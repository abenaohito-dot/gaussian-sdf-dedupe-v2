"""Conservative conformer comparisons. No coordinate mutation; no reflection."""
from dataclasses import dataclass, asdict
from itertools import permutations, product
import json
import math
import time
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


def torsions_array(x):
    """Proper torsions in degrees for an (n, 4, 3) coordinate array."""
    if not len(x): return np.empty(0)
    b0=x[:,0]-x[:,1]; b1=x[:,2]-x[:,1]; b2=x[:,3]-x[:,2]
    lengths=np.linalg.norm(b1,axis=1)
    if np.any(lengths<1e-10): raise ValueError('degenerate torsion')
    b1=b1/lengths[:,None]
    v=b0-np.sum(b0*b1,axis=1)[:,None]*b1
    w=b2-np.sum(b2*b1,axis=1)[:,None]*b1
    if np.any(np.linalg.norm(v,axis=1)<1e-8) or np.any(np.linalg.norm(w,axis=1)<1e-8):
        raise ValueError('undefined torsion; manual review')
    return np.degrees(np.arctan2(np.sum(np.cross(b1,v)*w,axis=1),np.sum(v*w,axis=1)))


def prepare_molecule(m):
    """Read-only geometry cache scoped to one run, never reused after edits."""
    if not explicit_complete(m): return {'valid':False}
    h,indices=heavy_copy(m)
    xyz=m.GetConformer().GetPositions()
    quads=set()
    for bond in m.GetBonds():
        j,k=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if m.GetAtomWithIdx(j).GetAtomicNum()==1 or m.GetAtomWithIdx(k).GetAtomicNum()==1: continue
        for ni,nl in product(m.GetAtomWithIdx(j).GetNeighbors(),m.GetAtomWithIdx(k).GetNeighbors()):
            i,l=ni.GetIdx(),nl.GetIdx()
            if len({i,j,k,l})!=4: continue
            heavy=ni.GetAtomicNum()>1 and nl.GetAtomicNum()>1
            hydroxyl=(ni.GetAtomicNum()==1 and m.GetAtomWithIdx(j).GetAtomicNum()==8) or (nl.GetAtomicNum()==1 and m.GetAtomWithIdx(k).GetAtomicNum()==8)
            if heavy or hydroxyl: quads.add(min((i,j,k,l),(l,k,j,i)))
    q=np.array(sorted(quads),dtype=int).reshape((-1,4))
    hs={i:tuple(n.GetIdx() for n in m.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum()==1) for i in indices}
    oh=np.array([(i,k) for i in indices if m.GetAtomWithIdx(i).GetAtomicNum()==8 for k in hs[i]],dtype=int).reshape((-1,2))
    rings=[tuple(r) for r in m.GetRingInfo().AtomRings() if len(r)==4 and all(m.GetAtomWithIdx(i).GetAtomicNum()==6 for i in r)]
    return dict(valid=True,heavy=h,indices=np.array(indices,dtype=int),xyz=xyz,
                smiles=Chem.MolToSmiles(m,isomericSmiles=True),quads=q,torsions=torsions_array(xyz[q]),
                hydrogens=hs,isotopes=[a.GetIsotope() for a in m.GetAtoms()],oh=oh,rings=rings)


def heavy_alignment(pa,pb,match):
    targets=pb['indices'][list(match)]
    r,t=fit(pa['xyz'][pa['indices']],pb['xyz'][targets])
    aligned=pa['xyz']@r+t
    rms=float(np.sqrt(np.mean(np.sum((aligned[pa['indices']]-pb['xyz'][targets])**2,axis=1))))
    return aligned,targets,rms


def mapped_metrics(a,b,ha,hb,ia,ib,match,prepared=None,alignment=None):
    pa,pb=prepared if prepared is not None else (prepare_molecule(a),prepare_molecule(b))
    xa,xb=pa['xyz'],pb['xyz']
    aligned,targets,rms=alignment if alignment is not None else heavy_alignment(pa,pb,match)
    mapping=np.full(a.GetNumAtoms(),-1,dtype=int)
    mapping[pa['indices']]=targets
    for i,j in zip(pa['indices'],targets):
        ah,bh=pa['hydrogens'][i],pb['hydrogens'][j]
        if len(ah)!=len(bh) or len(ah)>6: raise ValueError('H correspondence incomplete')
        if not ah: continue
        choices=[p for p in permutations(bh) if all(pa['isotopes'][k]==pb['isotopes'][l] for k,l in zip(ah,p))]
        if not choices: raise ValueError('H isotope mismatch')
        choices=np.array(choices,dtype=int)
        costs=np.sum((aligned[list(ah)][None,:,:]-xb[choices])**2,axis=(1,2))
        mapping[list(ah)]=choices[np.argmin(costs)]
    if np.any(mapping<0) or len(set(mapping.tolist()))!=b.GetNumAtoms(): raise ValueError('incomplete mapping')
    oh_max=0.
    if len(pa['oh']):
        o,h=pa['oh'].T; v=aligned[h]-aligned[o]; w=xb[mapping[h]]-xb[mapping[o]]
        lengths=np.linalg.norm(v,axis=1)*np.linalg.norm(w,axis=1)
        if np.any(lengths<1e-10): raise ValueError('zero-length OH bond')
        oh_max=float(np.max(np.degrees(np.arccos(np.clip(np.sum(v*w,axis=1)/lengths,-1,1)))))
    mapped_torsions=torsions_array(xb[mapping[pa['quads']]])
    di=np.abs((pa['torsions']-mapped_torsions+180)%360-180)
    ring_di=[]; ring_h=[]
    for ring in pa['rings']:
        x,y=xa[list(ring)],xb[mapping[list(ring)]]
        ring_h.append(float(np.max(np.abs(height(x)-height(y)))))
        for k in range(4):
            q=[(k+j)%4 for j in range(4)]
            ring_di.append(periodic(torsion(x[q]),torsion(y[q])))
    return dict(heavy_rmsd=rms,
                all_atom_rmsd=float(np.sqrt(np.mean(np.sum((aligned-xb[mapping])**2,axis=1)))),
                oh_max_deg=oh_max,torsion_max_deg=float(np.max(di)) if len(di) else 0.,
                ring_max_deg=max(ring_di,default=0.),ring_height_delta=max(ring_h,default=0.),
                oh_count=len(pa['oh']),cyclobutane_count=len(ring_h),
                atom_map_1based={str(i+1):int(j)+1 for i,j in enumerate(mapping)})


def compare(a,b,cfg=Thresholds(),symmetry=True,prepared=None):
    base={'decision':HOLD,'reasons':[], 'metrics':{},'maps_examined':0,'mapping_truncated':False}
    try:
        pa,pb=prepared if prepared is not None else (prepare_molecule(a),prepare_molecule(b))
        if 'error' in pa or 'error' in pb:
            base['reasons']=['undefined geometry: '+str(pa.get('error') or pb.get('error'))]; return base
        if not pa['valid'] or not pb['valid']:
            base['reasons']=['explicit H or valid 3D coordinates missing']; return base
        # Full graph equality closes substructure wildcard behavior for isotopes/charges.
        if pa['smiles'] != pb['smiles']:
            base['reasons']=['molecular graph, isotope, charge or stereochemistry differ']; return base
        ha,ia=pa['heavy'],pa['indices']; hb,ib=pb['heavy'],pb['indices']
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
        candidates=[]; errors=0; screened=0
        for match in matches:
            try:
                alignment=heavy_alignment(pa,pb,match)
                if alignment[2] > cfg.heavy*cfg.separate_factor:
                    # This mapping already fails a separation band. No full-atom
                    # computation can turn it into a confirmed duplicate.
                    candidates.append({'heavy_rmsd':alignment[2],'heavy_screened':True})
                    screened+=1
                else:
                    candidates.append(mapped_metrics(a,b,ha,hb,ia,ib,match,(pa,pb),alignment))
            except ValueError: errors+=1
        base['maps_screened_by_heavy_rmsd']=screened
        base['maps_examined']=len(matches)
        if not candidates: base['reasons']=['all mappings have undefined geometry']; return base
        def score(m): return max(m[k]/v for k,v in limits.items() if k in m)
        witness=min(candidates,key=score)
        best=witness if score(witness)<=1 else min(candidates,key=lambda m:m['heavy_rmsd'])
        base['metrics']=best
        base['minimum_combined_score_lower_bound' if screened else 'minimum_combined_score']=score(witness)
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
        base['exceeded']=[k for k,v in limits.items() if k in best and best[k]>v]
        if best.get('heavy_screened'):
            base['reasons'].append('heavy-atom separation band exceeded; detailed metrics not evaluated')
        return base
    except Exception as e:
        base['reasons']=[f'comparison failed: {type(e).__name__}: {e}']; return base


def deduplicate(mols,cfg=Thresholds(),symmetry=True,progress_callback=None,evidence_mode="all"):
    if evidence_mode not in ("all","representatives"): raise ValueError("unknown evidence mode")
    started=time.monotonic(); last_update=started; completed=0; pair_count=0
    def report(label='',force=False):
        nonlocal last_update
        now=time.monotonic()
        if progress_callback and (force or now-last_update>=0.25):
            progress_callback(completed,len(mols),pair_count,label,now-started)
            last_update=now
    report(force=True)
    prepared={}
    for m in mols:
        try: prepared[id(m)]=prepare_molecule(m)
        except Exception as e: prepared[id(m)]={'valid':False,'error':str(e)}
    groups={}
    for m in mols: groups.setdefault(energy_group(m),[]).append(m)
    kept=[]; rows=[]; pairs=[]
    for group_id,group in enumerate(groups.values(),1):
        ordered=sorted(group,key=lambda m:energy(m) if energy(m) is not None else float('inf'))
        e0=min((energy(m) for m in group if energy(m) is not None),default=None)
        for rank,m in enumerate(ordered,1):
            label=prop(m,'SourceLabel') or f'mol_{len(rows)+1}'
            best_hold=None; best_distinct=None; duplicate=None; item_comparisons=0
            for ref in kept:
                report(label)
                result=compare(m,ref,cfg,symmetry,prepared=(prepared[id(m)],prepared[id(ref)]))
                pair_count+=1
                record={'source_label':label,'reference':prop(ref,'SourceLabel'),'group':group_id,**result}
                item_comparisons+=1
                if evidence_mode=='all': pairs.append(record)
                if result['decision']==HOLD:
                    if best_hold is None or record['metrics'].get('heavy_rmsd',float('inf')) < best_hold['metrics'].get('heavy_rmsd',float('inf')): best_hold=record
                elif result['decision']==DIFFERENT:
                    if best_distinct is None or record['metrics'].get('heavy_rmsd',float('inf')) < best_distinct['metrics'].get('heavy_rmsd',float('inf')): best_distinct=record
                if result['decision']==DUP: duplicate=record; break
            if duplicate: status=DUP; chosen=duplicate
            elif best_hold is not None: status=HOLD; chosen=best_hold
            elif best_distinct is not None: status=DIFFERENT; chosen=best_distinct
            else: status=HOLD; chosen={'reasons':['first representative; no pair comparison'],'metrics':{}}
            if evidence_mode=='representatives' and item_comparisons: pairs.append(chosen)
            if not duplicate: kept.append(m)
            m.SetProp('V2Decision',status); m.SetProp('V2Reason','; '.join(chosen['reasons']))
            scale={'Hartree':627.509474,'kcal/mol':1.,'kJ/mol':1/4.184}.get(prop(m,'AppEnergyUnit'))
            row=dict(source_label=label,source_file=prop(m,'SourceFile'),input_type=prop(m,'InputType'),record_index=prop(m,'RecordIndex'),
                     energy_type=prop(m,'AppEnergyType'),energy_property=prop(m,'AppEnergySource'),energy_unit=prop(m,'AppEnergyUnit'),energy_value=energy(m),
                     energy_group=group_id,relative_energy_kcal_mol=((energy(m)-e0)*scale if energy(m) is not None and scale else None),rank_by_energy=rank,
                     comparisons_evaluated=item_comparisons,status=status,kept=not bool(duplicate),duplicate_of=duplicate['reference'] if duplicate else '',
                     reason='; '.join(chosen['reasons']),normal_termination=prop(m,'NormalTermination'),optimization_verified=prop(m,'OptimizationVerified'),
                     **{k:v for k,v in chosen['metrics'].items() if k!='atom_map_1based'})
            row['rmsd_to_representative']=row.get('heavy_rmsd')
            rows.append(row)
            completed+=1
            report(label,force=True)
    return kept,rows,pairs
