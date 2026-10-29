"""Conservative single-job Gaussian provenance and conversion checks."""
import re
import json
import numpy as np
from engine import fit

NUM=r'[-+]?\d+(?:\.\d*)?(?:[DEde][-+]?\d+)?'
def number(s): return float(s.replace('D','E').replace('d','e'))

def inspect_log(text):
    text=text.replace('\r','')
    routes=re.findall(r'^\s*#([^\n]*(?:\n(?!\s*-{3,})[^\n]*)*)',text,re.M)
    # Only echoed route sections start with '#'; archive route is mid-line.
    route=' '.join(routes[-1].lower().split()) if routes else ''
    normal=text.rfind('Normal termination of Gaussian')
    error=text.rfind('Error termination')
    completed=text.rfind('Optimization completed')
    failed=max(text.rfind('Optimization stopped'),text.rfind('Number of steps exceeded'),text.rfind('Convergence failure'))
    tables=list(re.finditer(r'Maximum Force\s+('+NUM+r')\s+('+NUM+r')\s+(YES|NO)\s*\n\s*RMS\s+Force\s+('+NUM+r')\s+('+NUM+r')\s+(YES|NO)\s*\n\s*Maximum Displacement\s+('+NUM+r')\s+('+NUM+r')\s+(YES|NO)\s*\n\s*RMS\s+Displacement\s+('+NUM+r')\s+('+NUM+r')\s+(YES|NO)',text))
    flags=[tables[-1].group(i) for i in (3,6,9,12)] if tables else []
    converged=(len(routes)==1 and bool(re.search(r'\bopt\b',route)) and completed>failed and normal>completed>=0 and normal>error
               and flags==['YES']*4 and tables[-1].start()<completed)
    frequencies=[number(v) for line in re.findall(r'Frequencies --([^\n]+)',text) for v in re.findall(NUM,line)]
    scf=re.findall(r'SCF Done:\s+E\([^)]*\)\s*=\s*('+NUM+')',text)
    gibbs=re.findall(r'Sum of electronic and thermal Free Energies\s*=\s*('+NUM+')',text)
    if not scf:
        scf=re.findall(r'[|\\]HF=('+NUM+')',text)
    charge=re.findall(r'Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)',text)
    orientations=[]
    for match in re.finditer(r'(?:Standard|Input) orientation:',text):
        tail=text[match.end():]; parts=re.split(r'\n\s*-{5,}[^\n]*\n',tail,maxsplit=3)
        if len(parts)<3: continue
        rows=[]
        for line in parts[2].splitlines():
            vals=line.split()
            if len(vals)==6 and vals[0].isdigit():
                try: rows.append((int(vals[1]),[number(v) for v in vals[3:6]]))
                except ValueError: pass
        if rows: orientations.append(rows)
    return dict(route=route,route_count=len(routes),normal_termination=normal>=0 and normal>error,
                optimization_converged=converged,convergence_flags=flags,imaginary_frequencies=sum(v<0 for v in frequencies),
                frequencies_present=bool(frequencies),scf=number(scf[-1]) if scf else None,gibbs=number(gibbs[-1]) if gibbs else None,
                charge_multiplicity=charge[-1] if charge else None,orientation=orientations[-1] if orientations else [])


def annotate_gaussian(mol,path):
    audit=inspect_log(path.read_text(errors='replace'))
    orient=audit.pop('orientation'); ok=False; residual=None
    if len(orient)==mol.GetNumAtoms() and [a.GetAtomicNum() for a in mol.GetAtoms()]==[row[0] for row in orient]:
        x=mol.GetConformer().GetPositions(); y=np.array([row[1] for row in orient]); r,t=fit(x,y)
        residual=float(np.max(np.linalg.norm(x@r+t-y,axis=1))); ok=residual<0.002
    audit['conversion_max_residual_A']=residual
    audit['geometry_verified']=ok
    mol.SetProp('OptimizationVerified',str(audit['optimization_converged'] and ok))
    mol.SetProp('NormalTermination',str(audit['normal_termination']))
    mol.SetProp('CalculationContext',json.dumps([audit['route'],audit['charge_multiplicity']],sort_keys=True))
    mol.SetProp('ImaginaryFrequencies',str(audit['imaginary_frequencies']))
    mol.SetProp('GaussianAudit',json.dumps(audit,ensure_ascii=False))
    kind=mol.GetProp('AppEnergyType') if mol.HasProp('AppEnergyType') else 'SCF'
    val=audit['scf'] if kind=='SCF' else audit['gibbs']
    if mol.HasProp('AppEnergy'): mol.ClearProp('AppEnergy')
    if val is not None: mol.SetProp('AppEnergy',str(val))
    return audit
