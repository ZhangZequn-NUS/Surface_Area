#!/usr/bin/env python3
"""Export Origin-ready AlFu/N2 density maps and framework-distance curves.

Example:
 python analyze_alfu_origin.py --cif 'AlFumarate_Charge2(1).cif' --movies 'MOVIE_~1.PDB' 'MOVIE_~2.PDB' 'MOVIE_~3.PDB' 'MOVIE_~4.PDB' --out Origin_results

Each 2D projection is an average of 3D number density along the omitted fractional
axis (molecules/nm^3). Values are grid-cell-centered; framework overlays are
fractional positions, NOT projected physical contacts. Distances use molecular
midpoints and the nearest framework atomic center with periodic images.
"""
import argparse,csv,re,shlex
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def cell_matrix(a,b,c,alpha,beta,gamma):
    al,be,ga=np.deg2rad([alpha,beta,gamma]); ca,cb,cg=np.cos([al,be,ga]);sg=np.sin(ga)
    va=np.array([a,0.,0.]);vb=np.array([b*cg,b*sg,0.]);vc=np.array([c*cb,c*(ca-cb*cg)/sg,0.]);vc[2]=np.sqrt(max(c*c-np.dot(vc[:2],vc[:2]),0.))
    return np.column_stack([va,vb,vc])


def movie_frames(path):
    atoms=[];cell=None;frame=0
    with open(path,encoding='utf-8',errors='replace') as f:
        for line in f:
            tag=line[:6].strip()
            if tag=='MODEL':atoms=[]
            elif tag=='CRYST1':cell=cell_matrix(*[float(line[s:e]) for s,e in ((6,15),(15,24),(24,33),(33,40),(40,47),(47,54))])
            elif tag in ('ATOM','HETATM'):atoms.append([float(line[s:e]) for s,e in ((30,38),(38,46),(46,54))])
            elif tag=='ENDMDL':
                frame+=1
                if cell is None or len(atoms)%2:raise ValueError(f'Invalid frame {frame} in {path}')
                yield frame,cell,np.array(atoms).reshape(-1,2,3)
                atoms=[]


def read_cif(path):
    lines=Path(path).read_text().splitlines();params={};syms=[];atoms=[];i=0
    for line in lines:
        z=line.split()
        if len(z)>=2 and z[0].startswith('_cell_'):
            try:params[z[0]]=float(re.sub(r'\([^)]*\)$','',z[1]))
            except ValueError:pass
    cell=cell_matrix(*[params['_cell_length_'+x] for x in 'abc'],*[params['_cell_angle_'+x] for x in ('alpha','beta','gamma')])
    while i<len(lines):
        if lines[i].strip()!='loop_':i+=1;continue
        i+=1;headers=[]
        while i<len(lines) and lines[i].strip().startswith('_'):
            headers.append(lines[i].strip());i+=1
        data=[]
        while i<len(lines):
            line=lines[i].strip()
            if line=='loop_' or line.startswith('_'):break
            if line and not line.startswith('#'):data.append(shlex.split(line))
            i+=1
        for key in ('_symmetry_equiv_pos_as_xyz','_space_group_symop_operation_xyz'):
            if key in headers:syms=[row[headers.index(key)] for row in data]
        if '_atom_site_fract_x' in headers:
            ids=[headers.index('_atom_site_'+v) for v in ('type_symbol','fract_x','fract_y','fract_z')]
            for row in data:atoms.append((row[ids[0]],np.array([float(re.sub(r'\([^)]*\)$','',row[k])) for k in ids[1:]])))
    if not syms:syms=['x,y,z']
    expanded=[]
    for element,xyz in atoms:
        for sym in syms:
            env=dict(zip('xyz',xyz))
            point=np.array([eval(expr,{'__builtins__':{}},env) for expr in sym.replace(' ','').split(',')],float)%1
            if not any(el==element and np.linalg.norm((p-point+.5)%1-.5)<1e-5 for el,p in expanded):expanded.append((element,point))
    if not expanded:raise ValueError('No framework atoms parsed from CIF')
    return cell,expanded


def csvwrite(path,header,rows):
    with open(path,'w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)


def run(movie,unit,framework,out,n,sigma,stride,rep,bins):
    out.mkdir(parents=True,exist_ok=True)
    grid=np.zeros((n,n,n),float);points=[];counts=[];moviecell=None
    for frame,cell,pairs in movie_frames(movie):
        if (frame-1)%stride:continue
        if moviecell is None:moviecell=cell
        if not np.allclose(moviecell,cell,atol=.06):raise ValueError('Movie cell changes')
        frac=np.einsum('ij,nkj->nki',np.linalg.inv(cell),pairs)
        delta=(frac[:,1]-frac[:,0]+.5)%1-.5
        mid=(frac[:,0]+.5*delta)%1
        folded=(mid*np.array(rep))%1
        points.append(folded);counts.append((frame,len(folded)))
        ix=np.minimum((folded*n).astype(int),n-1)
        np.add.at(grid,tuple(ix.T),1)
    if not points:raise ValueError('No movie frames')
    if not np.allclose(unit@np.diag(rep),moviecell,atol=.07):raise ValueError('CIF/movie cell mismatch')
    vol=abs(np.linalg.det(unit))/1000
    den=grid/(len(points)*np.prod(rep)*vol/n**3)
    sm=gaussian_filter(den,sigma,mode='wrap') if sigma else den.copy()
    centers=(np.arange(n)+.5)/n
    np.savez_compressed(out/'density_3d.npz',raw=den,smoothed=sm,cell_angstrom=unit,centers=centers)
    csvwrite(out/'density_3d_origin.csv',['fractional_a','fractional_b','fractional_c','raw_N2_per_nm3','smoothed_N2_per_nm3'],
             ((centers[i],centers[j],centers[k],den[i,j,k],sm[i,j,k]) for i in range(n) for j in range(n) for k in range(n)))
    for i,j,k,label in [(0,1,2,'ab'),(0,2,1,'ac'),(1,2,0,'bc')]:
        # mean over omitted axis; matrix rows correspond y, columns correspond x
        raw=den.mean(axis=k); smooth=sm.mean(axis=k)
        csvwrite(out/f'projection_{label}_XYZ.csv',[f'fractional_{"abc"[i]}',f'fractional_{"abc"[j]}','raw_N2_per_nm3','smoothed_N2_per_nm3'],
                 ((centers[x],centers[y],raw[x,y],smooth[x,y]) for x in range(n) for y in range(n)))
        csvwrite(out/f'projection_{label}_matrix.csv',[f'{"abc"[j]}_rows__{"abc"[i]}_columns']+[f'{x:.8f}' for x in centers],
                 ((f'{centers[y]:.8f}',)+tuple(smooth[:,y]) for y in range(n)))
    for axis,name in enumerate('abc'):
        raw=den.mean(axis=tuple(j for j in range(3) if j!=axis))
        smooth=sm.mean(axis=tuple(j for j in range(3) if j!=axis))
        csvwrite(out/f'profile_{name}.csv',[f'fractional_{name}','raw_N2_per_nm3','smoothed_N2_per_nm3'],zip(centers,raw,smooth))
    fwfrac=np.array([p for _,p in framework]);els=[e for e,_ in framework]
    csvwrite(out/'framework_overlay_fractional.csv',['element','fractional_a','fractional_b','fractional_c'],((e,*p) for e,p in framework))
    allfrac=np.concatenate(points)
    fwcart=(unit@fwfrac.T).T
    images=np.concatenate([fwcart+unit@np.array([i,j,k]) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)])
    d,_=cKDTree(images).query((unit@allfrac.T).T)
    edges=np.arange(0,max(bins,d.max()+bins)+bins,bins)
    hist,_=np.histogram(d,bins=edges)
    csvwrite(out/'framework_distance_histogram.csv',['distance_center_A','probability_density_per_A','count','fraction_in_bin'],
             (((lo+hi)/2,h/(len(d)*(hi-lo)),int(h),h/len(d)) for lo,hi,h in zip(edges[:-1],edges[1:],hist)))
    csvwrite(out/'frame_counts.csv',['frame','number_N2'],counts)
    csvwrite(out/'N2_centers_folded.csv',['frame','fractional_a','fractional_b','fractional_c'],
             ((counts[ii][0],*p) for ii,arr in enumerate(points) for p in arr))
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    projections=[sm.mean(axis=2),sm.mean(axis=1),sm.mean(axis=0)]
    vmax=max(p.max() for p in projections)
    for ax,img,(i,j) in zip(axs,projections,[(0,1),(0,2),(1,2)]):
        im=ax.imshow(img.T,origin='lower',extent=(0,1,0,1),vmin=0,vmax=vmax,cmap='viridis')
        for e in sorted(set(els)):
            p=fwfrac[np.array([x==e for x in els])]
            ax.scatter(p[:,i],p[:,j],s=10 if e=='Al' else 4,alpha=.75,label=e)
        ax.set(xlabel='fractional '+ 'abc'[i],ylabel='fractional '+'abc'[j],xlim=(0,1),ylim=(0,1))
    fig.colorbar(im,ax=axs,label='N2 molecules / nm3')
    fig.savefig(out/'overlay_preview.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.2,3.5));ax.stairs(hist/(len(d)*np.diff(edges)),edges)
    ax.set(xlabel='N2 midpoint to nearest framework atom (A)',ylabel='Probability density (1/A)')
    fig.tight_layout();fig.savefig(out/'distance_preview.png',dpi=180);plt.close(fig)
    return {'movie':movie.name,'frames':len(points),'mean_N2_per_supercell':float(np.mean([v for _,v in counts])),
            'mean_N2_per_unit_cell':float(np.mean([v for _,v in counts])/np.prod(rep)),
            'mean_N2_framework_distance_A':float(d.mean()),'grid':n,'sigma_grid_cells':sigma,'unit_cell_volume_nm3':vol}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--movies',nargs='+',type=Path,required=True)
    ap.add_argument('--cif',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('Origin_results'))
    ap.add_argument('--grid',type=int,default=48)
    ap.add_argument('--sigma',type=float,default=1.0)
    ap.add_argument('--stride',type=int,default=1)
    ap.add_argument('--replication',type=int,nargs=3,default=[3,3,3])
    ap.add_argument('--distance-bin',type=float,default=.25)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    unit,fw=read_cif(a.cif);results=[]
    for movie in a.movies:
        folder=a.out/movie.stem
        print('Processing',movie)
        results.append(run(movie,unit,fw,folder,a.grid,a.sigma,a.stride,a.replication,a.distance_bin))
    csvwrite(a.out/'summary.csv',list(results[0]),([r[k] for k in results[0]] for r in results))
    print('Finished:',a.out)

if __name__=='__main__':main()
