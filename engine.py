"""Conservative conformer comparisons. No coordinate mutation; no reflection."""
from dataclasses import dataclass, asdict
from itertools import permutations, product
import json
import math
import numpy as np
from rdkit import Chem

DUP, DIFFERENT, HOLD = 'Confirmed duplicate', 'Distinct conformer', 'Needs review'

@dataclass(frozen=True)
class Thresholds:
    # Provisional screening/confirmation bands, NOT calibrated chemical constants.
    heavy: float = 0.20
    all_atom: float = 0.08
    oh_deg: float = 12.0
    torsion_deg: float = 8.0
    ring_deg: float = 3.0
    ring_height: float = 0.015
    energy_kcal: float = 0.05
    separate_factor: float = 2.0
    max_maps: int = 4096

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0: raise ValueError(f'{key} must be finite and positive')
        if self.separate_factor <= 1: raise ValueError('separate_factor must exceed 1')
        if not isinstance(self.max_maps, int): raise ValueError('max_maps must be an integer')


def prop(m, key, default=''):
    return m.GetProp(key) if m.HasProp(key) else default


def energy(m):
    try:
        e = float(prop(m, 'AppEnergy'))
        return e if math.isfinite(e) else None
    except ValueError:
        return None


def energy_group(m):
    # A unit match alone never authorizes cross-method ranking/comparison.
    return tuple(prop(m, x) for x in ('InputType','AppEnergyType','AppEnergyUnit','AppEnergySource','CalculationContext'))


def delta_energy(a, b):
    ea, eb = energy(a), energy(b)
    if ea is None or eb is None or energy_group(a) != energy_group(b): return None
    scale = {'Hartree':627.509474, 'kcal/mol':1., 'kJ/mol':1/4.184}.get(prop(a,'AppEnergyUnit'))
    return abs(ea-eb)*scale if scale else None


def fit(x,y):
    xc, yc = x.mean(axis=0), y.mean(axis=0)
    u, _, vt = np.linalg.svd((x-xc).T@(y-yc))
    d = np.eye(3); d[-1,-1] = np.linalg.det(u@vt)
    r = u@d@vt
    return r, yc-xc@r


def angle(x,y):
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if min(nx,ny)<1e-10: raise ValueError('zero-length bond')
    return float(np.degrees(np.arccos(np.clip(x@y/(nx*ny),-1,1))))


def torsion(x):
    b0=x[0]-x[1]; b1=x[2]-x[1]; b2=x[3]-x[2]
    n=np.linalg.norm(b1)
    if n<1e-10: raise ValueError('degenerate torsion')
    b1=b1/n; v=b0-(b0@b1)*b1; w=b2-(b2@b1)*b1
    if min(np.linalg.norm(v),np.linalg.norm(w))<1e-8: raise ValueError('linear torsion')
    return float(np.degrees(np.arctan2(np.cross(b1,v)@w,v@w)))


def periodic(a,b): return abs((a-b+180)%360-180)


def height(x):
    # Ordered ring fixes normal sign. Same mapped order is used in both molecules.
    normal=np.cross(x[2]-x[0],x[3]-x[1]); n=np.linalg.norm(normal)
    if n<1e-10: raise ValueError('degenerate ring')
    return (x-x.mean(axis=0))@(normal/n)


def explicit_complete(m):
    return (m.GetNumConformers()>0 and m.GetConformer().Is3D()
            and all(a.GetNumImplicitHs()==0 and a.GetNumExplicitHs()==0 for a in m.GetAtoms())
            and np.isfinite(m.GetConformer().GetPositions()).all())


def heavy_copy(m):
    c=Chem.Mol(m)
    for a in c.GetAtoms(): a.SetIntProp('_OriginalIndex',a.GetIdx())
    h=Chem.RemoveHs(c)
    return h,[a.GetIntProp('_OriginalIndex') for a in h.GetAtoms()]


def mapped_metrics(a,b,ha,hb,ia,ib,match):
    xa,xb=a.GetConformer().GetPositions(),b.GetConformer().GetPositions()
    mapping={ia[i]:ib[j] for i,j in enumerate(match)}
    inds=list(mapping); targets=[mapping[i] for i in inds]
    r,t=fit(xa[inds],xb[targets]); aligned=xa@r+t
    # Hydrogens may permute ONLY within the corresponding bonded atom, isotope preserved.
    for i,j in list(mapping.items()):
        ah=[n.GetIdx() for n in a.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum()==1]
        bh=[n.GetIdx() for n in b.GetAtomWithIdx(j).GetNeighbors() if n.GetAtomicNum()==1]
        if len(ah)!=len(bh) or len(ah)>6: raise ValueError('H correspondence incomplete')
        choices=[p for p in permutations(bh) if all(a.GetAtomWithIdx(k).GetIsotope()==b.GetAtomWithIdx(l).GetIsotope() for k,l in zip(ah,p))]
        if not choices: raise ValueError('H isotope mismatch')
        best=min(choices,key=lambda p:sum(np.sum((aligned[k]-xb[l])**2) for k,l in zip(ah,p)))
        mapping.update(zip(ah,best))
    if len(mapping)!=a.GetNumAtoms() or len(set(mapping.values()))!=b.GetNumAtoms(): raise ValueError('incomplete mapping')
    oh=[]; di=[]; ring_di=[]; ring_h=[]
    for i in inds:
        if a.GetAtomWithIdx(i).GetAtomicNum()==8:
            for n in a.GetAtomWithIdx(i).GetNeighbors():
                if n.GetAtomicNum()==1:
                    k=n.GetIdx(); oh.append(angle(aligned[k]-aligned[i],xb[mapping[k]]-xb[mapping[i]]))
    # All heavy-atom proper torsions, including rings, plus all OH torsions.
    quads=set()
    for bond in a.GetBonds():
        j,k=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if a.GetAtomWithIdx(j).GetAtomicNum()==1 or a.GetAtomWithIdx(k).GetAtomicNum()==1: continue
        for ni,nl in product(a.GetAtomWithIdx(j).GetNeighbors(),a.GetAtomWithIdx(k).GetNeighbors()):
            i,l=ni.GetIdx(),nl.GetIdx()
            if len({i,j,k,l})!=4: continue
            heavy=ni.GetAtomicNum()>1 and nl.GetAtomicNum()>1
            hydroxyl=(ni.GetAtomicNum()==1 and a.GetAtomWithIdx(j).GetAtomicNum()==8) or (nl.GetAtomicNum()==1 and a.GetAtomWithIdx(k).GetAtomicNum()==8)
            if heavy or hydroxyl: quads.add(min((i,j,k,l),(l,k,j,i)))
    for q in quads:
        try: di.append(periodic(torsion(xa[list(q)]),torsion(xb[[mapping[i] for i in q]])))
        except ValueError: raise ValueError('undefined torsion; manual review')
    for ring in a.GetRingInfo().AtomRings():
        if len(ring)!=4 or not all(a.GetAtomWithIdx(i).GetAtomicNum()==6 for i in ring): continue
        x,y=xa[list(ring)],xb[[mapping[i] for i in ring]]
        ring_h.append(float(np.max(np.abs(height(x)-height(y)))))
        for k in range(4):
            q=[(k+j)%4 for j in range(4)]
            ring_di.append(periodic(torsion(x[q]),torsion(y[q])))
    return dict(heavy_rmsd=float(np.sqrt(np.mean(np.sum((aligned[inds]-xb[targets])**2,axis=1)))),
                all_atom_rmsd=float(np.sqrt(np.mean([np.sum((aligned[i]-xb[j])**2) for i,j in mapping.items()]))),
                oh_max_deg=max(oh,default=0.),torsion_max_deg=max(di,default=0.),
                ring_max_deg=max(ring_di,default=0.),ring_height_delta=max(ring_h,default=0.),
                oh_count=len(oh),cyclobutane_count=len(ring_h),
                atom_map_1based={str(i+1):j+1 for i,j in mapping.items()})


def compare(a,b,cfg=Thresholds(),symmetry=True):
    base={'decision':HOLD,'reasons':[], 'metrics':{},'maps_examined':0,'mapping_truncated':False}
    try:
        if not explicit_complete(a) or not explicit_complete(b):
            base['reasons']=['explicit H or valid 3D coordinates missing']; return base
        # Full graph equality closes substructure wildcard behavior for isotopes/charges.
        if Chem.MolToSmiles(a, isomericSmiles=True) != Chem.MolToSmiles(b, isomericSmiles=True):
            base['reasons']=['molecular graph, isotope, charge or stereochemistry differ']; return base
        ha,ia=heavy_copy(a); hb,ib=heavy_copy(b)
        if ha.GetNumAtoms()!=hb.GetNumAtoms() or ha.GetNumBonds()!=hb.GetNumBonds():
            base.update(decision=DIFFERENT,reasons=['different molecular graph']); return base
        matches=hb.GetSubstructMatches(ha,uniquify=False,useChirality=True,maxMatches=cfg.max_maps+1)
        if not symmetry: matches=tuple(m for m in matches if tuple(m)==tuple(range(len(ia))))
        truncated=len(matches)>cfg.max_maps; matches=matches[:cfg.max_maps]
        base['mapping_truncated']=truncated
        if not matches:
            base['reasons']=['no complete stereochemistry-preserving atom mapping']; return base
        limits=dict(heavy_rmsd=cfg.heavy,all_atom_rmsd=cfg.all_atom,oh_max_deg=cfg.oh_deg,
                    torsion_max_deg=cfg.torsion_deg,ring_max_deg=cfg.ring_deg,ring_height_delta=cfg.ring_height)
        candidates=[]; errors=0
        for match in matches:
            try: candidates.append(mapped_metrics(a,b,ha,hb,ia,ib,match))
            except ValueError: errors+=1
        base['maps_examined']=len(matches)
        if not candidates: base['reasons']=['all mappings have undefined geometry']; return base
        def score(m): return max(m[k]/v for k,v in limits.items())
        witness=min(candidates,key=score)
        best=witness if score(witness)<=1 else min(candidates,key=lambda m:m['heavy_rmsd'])
        base['metrics']=best
        base['minimum_combined_score']=score(witness)
        base['minimum_heavy_rmsd']=min(m['heavy_rmsd'] for m in candidates)
        de=delta_energy(a,b); base['energy_delta_kcal']=de
        quality=[]
        for m in (a,b):
            if prop(m,'InputType')=='log' and prop(m,'OptimizationVerified')!='True': quality.append('Gaussian optimization/geometry not verified')
            if prop(m,'ImaginaryFrequencies') and int(prop(m,'ImaginaryFrequencies'))>0: quality.append('imaginary frequencies present')
        if de is None: quality.append('energy missing or calculation contexts incomparable')
        elif de>cfg.energy_kcal: quality.append('energy difference exceeds confirmation band')
        # Cap/errors prevent asserting that every symmetry mapping is different.
        if score(witness)<=1 and not quality:
            base.update(decision=DUP,reasons=['one chemically valid mapping passes every confirmation band'])
        elif score(witness)>cfg.separate_factor and not truncated and not errors:
            base.update(decision=DIFFERENT,reasons=['every valid mapping exceeds a geometric separation band']+quality)
        else:
            base['reasons']=quality+(['mapping enumeration incomplete'] if truncated or errors else [])+['geometry in review band']
        base['exceeded']=[k for k,v in limits.items() if best[k]>v]
        return base
    except Exception as e:
        base['reasons']=[f'comparison failed: {type(e).__name__}: {e}']; return base


def deduplicate(mols,cfg=Thresholds(),symmetry=True):
    groups={}
    for m in mols: groups.setdefault(energy_group(m),[]).append(m)
    kept=[]; rows=[]; pairs=[]
    for group_id,group in enumerate(groups.values(),1):
        ordered=sorted(group,key=lambda m:energy(m) if energy(m) is not None else float('inf'))
        e0=min((energy(m) for m in group if energy(m) is not None),default=None)
        for rank,m in enumerate(ordered,1):
            label=prop(m,'SourceLabel') or f'mol_{len(rows)+1}'
            decisions=[]; duplicate=None
            for ref in kept:
                result=compare(m,ref,cfg,symmetry)
                record={'source_label':label,'reference':prop(ref,'SourceLabel'),'group':group_id,**result}
                pairs.append(record); decisions.append(record)
                if result['decision']==DUP: duplicate=record; break
            if duplicate: status=DUP; chosen=duplicate
            elif decisions:
                status=HOLD if any(d['decision']==HOLD for d in decisions) else DIFFERENT
                eligible=[d for d in decisions if d['decision']==status]
                chosen=min(eligible,key=lambda d:d.get('metrics',{}).get('heavy_rmsd',float('inf')))
            else: status=HOLD; chosen={'reasons':['first representative; no pair comparison'],'metrics':{}}
            if not duplicate: kept.append(m)
            m.SetProp('V2Decision',status); m.SetProp('V2Reason','; '.join(chosen['reasons']))
            scale={'Hartree':627.509474,'kcal/mol':1.,'kJ/mol':1/4.184}.get(prop(m,'AppEnergyUnit'))
            row=dict(source_label=label,source_file=prop(m,'SourceFile'),input_type=prop(m,'InputType'),record_index=prop(m,'RecordIndex'),
                     energy_type=prop(m,'AppEnergyType'),energy_property=prop(m,'AppEnergySource'),energy_unit=prop(m,'AppEnergyUnit'),energy_value=energy(m),
                     energy_group=group_id,relative_energy_kcal_mol=((energy(m)-e0)*scale if energy(m) is not None and scale else None),rank_by_energy=rank,
                     status=status,kept=not bool(duplicate),duplicate_of=duplicate['reference'] if duplicate else '',
                     reason='; '.join(chosen['reasons']),normal_termination=prop(m,'NormalTermination'),optimization_verified=prop(m,'OptimizationVerified'),
                     **{k:v for k,v in chosen['metrics'].items() if k!='atom_map_1based'})
            row['rmsd_to_representative']=row.get('heavy_rmsd')
            rows.append(row)
    return kept,rows,pairs
