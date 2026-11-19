"""Phase 6: integrated validation of Phase 4C models with Phase 5A bit-level AWGN.
Run: python phase6_final_validation.py
Requires: numpy, matplotlib, scikit-learn.
All model and margin choices use training/validation only. Test is evaluation only.
"""
from pathlib import Path
from math import erfc, sqrt
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT = Path(__file__).resolve().parent
CHANNEL_SEED = 10
BITS_SEED = 2026
SYMBOLS_PER_STEP = 4000
TARGET_BER = 1e-3
M_VALUES = (4, 16, 64)

# Exact channel construction, engineered features, split, and model search of Phase 4C.
rng_channel = np.random.default_rng(CHANNEL_SEED)
t = np.arange(1000)
channel = np.clip(12 + 7*np.sin(2*np.pi*t/40) + 1.5*rng_channel.normal(size=len(t)), 0, 25)
window = 5
X = np.array([channel[i-window:i] for i in range(window, len(channel))])
y = channel[window:]
train_end, valid_end = int(0.6*len(y)), int(0.8*len(y))

def features(x):
    return np.column_stack([x, x[:,-1]-x[:,-2], x[:,-2]-x[:,-3],
                            x[:,-1]-x[:,0], x.mean(axis=1), x.std(axis=1),
                            x.min(axis=1), x.max(axis=1)])

F = features(X)
Xtrain, Xval, Xtest = X[:train_end], X[train_end:valid_end], X[valid_end:]
Ftrain, Fval, Ftest = F[:train_end], F[train_end:valid_end], F[valid_end:]
ytrain, yval, ytest = y[:train_end], y[train_end:valid_end], y[valid_end:]

candidates = [('Original Random Forest', 'Original', RandomForestRegressor(
    n_estimators=150, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1), False)]
for depth in (6,10,None):
    for leaf in (1,3,5):
        candidates.append((f'Tuned RF depth={depth}, leaf={leaf}', 'Tuned RF',
            RandomForestRegressor(n_estimators=150, max_depth=depth,
                min_samples_leaf=leaf, random_state=42, n_jobs=-1), True))
for lr in (0.05,0.1):
    for leaves in (7,15):
        candidates.append((f'Gradient Boosting lr={lr}, leaves={leaves}', 'Gradient Boosting',
            HistGradientBoostingRegressor(learning_rate=lr, max_iter=200,
                max_leaf_nodes=leaves, min_samples_leaf=15,
                l2_regularization=1.0, random_state=42), True))

all_models = []
for name, family, model, engineered in candidates:
    model.fit(Ftrain if engineered else Xtrain, ytrain)
    pred_val = model.predict(Fval if engineered else Xval)
    pred_test = model.predict(Ftest if engineered else Xtest)
    all_models.append(dict(name=name, family=family, pred_val=pred_val,
        pred_test=pred_test, val_mae=mean_absolute_error(yval,pred_val),
        test_mae=mean_absolute_error(ytest,pred_test),
        test_rmse=np.sqrt(mean_squared_error(ytest,pred_test))))
selected_models = [all_models[0]]
for family in ('Tuned RF','Gradient Boosting'):
    selected_models.append(min((r for r in all_models if r['family']==family),
                               key=lambda r:r['val_mae']))

def estimated_ber(M, ebn0_db):
    k = int(np.log2(M))
    eb = 10**(float(ebn0_db)/10)
    return (2*(1-1/sqrt(M))/k)*erfc(sqrt(3*k*eb/(2*(M-1))))

def choose(pred, margin=0):
    return max((M for M in M_VALUES if estimated_ber(M,pred-margin)<=TARGET_BER), default=0)

def analytical_stats(pred, actual, margin):
    decisions = np.array([choose(p,margin) for p in pred],dtype=int)
    mask = decisions>0
    violations = sum(estimated_ber(int(M),float(db))>TARGET_BER
                     for M,db in zip(decisions[mask],actual[mask]))
    tx = int(mask.sum())
    return dict(margin=margin, avg_bits=float(np.log2(np.maximum(decisions,1)).mean()),
                tx=tx, violations=int(violations), rate=violations/tx if tx else float('nan'))

for r in selected_models:
    sweep = [(m,analytical_stats(r['pred_val'],yval,m)) for m in np.arange(0,6.01,.5)]
    feasible = [(m,s) for m,s in sweep if s['tx'] and s['rate']<=.05]
    if feasible:
        r['margin'] = max(feasible,key=lambda item:(item[1]['avg_bits'],-item[0]))[0]
    else:
        r['margin'] = min(sweep,key=lambda item:(item[1]['rate'],-item[1]['avg_bits']))[0]

# Same Gray-coded square-QAM modulator/demodulator and AWGN scaling as Phase 5A.
def gray_to_binary(gray):
    b = gray.copy()
    shift=1
    while shift<8:
        b ^= b>>shift
        shift*=2
    return b

def bits_to_symbols(bits,M):
    k=int(np.log2(M)); half=k//2; side=int(np.sqrt(M))
    pairs=bits.reshape(-1,k)
    weights=1<<np.arange(half-1,-1,-1)
    gi=pairs[:,:half]@weights; gq=pairs[:,half:]@weights
    i=gray_to_binary(gi.astype(np.int64)); q=gray_to_binary(gq.astype(np.int64))
    scale=np.sqrt(2*(M-1)/3)
    return ((2*i-side+1)+1j*(2*q-side+1))/scale

def symbols_to_bits(rx,M):
    k=int(np.log2(M)); half=k//2; side=int(np.sqrt(M))
    scale=np.sqrt(2*(M-1)/3)
    i=np.clip(np.rint((rx.real*scale+side-1)/2),0,side-1).astype(np.int64)
    q=np.clip(np.rint((rx.imag*scale+side-1)/2),0,side-1).astype(np.int64)
    gi=i^(i>>1); gq=q^(q>>1)
    shifts=np.arange(half-1,-1,-1)
    return np.concatenate([((gi[:,None]>>shifts)&1),
                           ((gq[:,None]>>shifts)&1)],axis=1).astype(np.uint8).ravel()

# Shared bit/noise realizations for all controllers at each step and M.
rng_bits=np.random.default_rng(BITS_SEED)
errors_by_M=np.zeros((len(ytest),len(M_VALUES)),dtype=np.int64)
bits_by_M=np.zeros_like(errors_by_M)
for step,actual_db in enumerate(ytest):
    for j,M in enumerate(M_VALUES):
        k=int(np.log2(M))
        bits=rng_bits.integers(0,2,SYMBOLS_PER_STEP*k,dtype=np.uint8)
        tx=bits_to_symbols(bits,M)
        ebn0=10**(actual_db/10)
        sigma=np.sqrt(1/(2*k*ebn0))
        noise=sigma*(rng_bits.standard_normal(SYMBOLS_PER_STEP)+
                     1j*rng_bits.standard_normal(SYMBOLS_PER_STEP))
        rx=symbols_to_bits(tx+noise,M)
        errors_by_M[step,j]=np.count_nonzero(bits!=rx)
        bits_by_M[step,j]=len(bits)

controllers=[]
for r in selected_models:
    for margin_label,margin in [('No margin',0.),('Validation margin',float(r['margin']))]:
        controllers.append((r['family']+' — '+margin_label,r['pred_test'],margin,r['test_mae']))
controllers.extend([('Last-value — no margin',Xtest[:,-1],0.,mean_absolute_error(ytest,Xtest[:,-1])),
                    ('Oracle — ideal',ytest,0.,0.)])

rows=[]
step_results={}
for name,pred,margin,mae in controllers:
    choices=np.array([choose(p,margin) for p in pred],dtype=int)
    errors=np.zeros(len(ytest),dtype=np.int64)
    sent=np.zeros(len(ytest),dtype=np.int64)
    for j,M in enumerate(M_VALUES):
        mask=choices==M
        errors[mask]=errors_by_M[mask,j]
        sent[mask]=bits_by_M[mask,j]
    tx=int(np.count_nonzero(choices))
    ber_steps=np.divide(errors,sent,out=np.full(len(sent),np.nan),where=sent>0)
    violations=int(np.count_nonzero(ber_steps>TARGET_BER))
    total_errors=int(errors.sum()); total_bits=int(sent.sum())
    theoretical=analytical_stats(pred,ytest,margin)
    rows.append(dict(controller=name,margin_db=margin,test_mae_db=mae,
        tx_steps=tx,avg_raw_bits_per_step=np.log2(np.maximum(choices,1)).mean(),
        transmitted_bits=total_bits,bit_errors=total_errors,
        measured_ber=total_errors/total_bits if total_bits else np.nan,
        measured_violation_steps=violations,
        measured_violation_rate=violations/tx if tx else np.nan,
        analytical_violation_steps=theoretical['violations'],
        correctly_recovered_bits_per_opportunity=(total_bits-total_errors)/(len(ytest)*SYMBOLS_PER_STEP)))
    step_results[name]=dict(choices=choices,ber=ber_steps)

print('PHASE 6 — CONSISTENT MODEL AND BIT-LEVEL VALIDATION')
print(f'Train={len(ytrain)}, validation={len(yval)}, test={len(ytest)}; symbols/step={SYMBOLS_PER_STEP}')
for r in selected_models:
    print(f"Selected {r['name']}; validation MAE={r['val_mae']:.4f}; test MAE={r['test_mae']:.4f}; margin={r['margin']:.1f} dB")
print('\nController | margin | raw bits/step | measured BER | measured violations | analytic violations')
for r in rows:
    print(f"{r['controller']:<34} {r['margin_db']:>4.1f} {r['avg_raw_bits_per_step']:>10.3f} "
          f"{r['measured_ber']:>13.6f} {r['measured_violation_steps']:>3}/{r['tx_steps']:<3} "
          f"{r['analytical_violation_steps']:>3}/{r['tx_steps']:<3}")

csv_path=OUT/'phase6_final_comparison.csv'
with csv_path.open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

labels=[r['controller'].replace(' — ','\n') for r in rows]
x=np.arange(len(rows))
fig,axes=plt.subplots(2,1,figsize=(13,9),constrained_layout=True)
axes[0].bar(x,[r['avg_raw_bits_per_step'] for r in rows])
axes[0].set_ylabel('Average raw bits / symbol opportunity')
axes[0].set_title('Capacity selected by each controller (No TX = 0)')
axes[0].set_xticks(x,labels,rotation=15,ha='right')
axes[0].grid(axis='y',alpha=.25)
axes[1].bar(x,[r['measured_ber'] for r in rows])
axes[1].axhline(TARGET_BER,linestyle='--',color='black',label='Target BER = 0.001')
axes[1].set_ylabel('Measured aggregate BER')
axes[1].set_title('Bit-level BER (different transmitted bit totals)')
axes[1].set_xticks(x,labels,rotation=15,ha='right')
axes[1].legend();axes[1].grid(axis='y',alpha=.25)
fig.savefig(OUT/'phase6_capacity_ber.png',dpi=170)
plt.close(fig)

fig2,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
width=.38
ax.bar(x-width/2,[100*r['measured_violation_rate'] for r in rows],width,label='Measured step violations (%)')
ax.bar(x+width/2,[100*r['analytical_violation_steps']/r['tx_steps'] for r in rows],width,label='Analytical step violations (%)')
ax.axhline(5,linestyle='--',color='black',label='Validation design goal: 5%')
ax.set_xticks(x,labels,rotation=15,ha='right')
ax.set_ylabel('Fraction of transmitting steps exceeding BER target (%)')
ax.set_title('Finite-bit measured vs analytical step violation rates')
ax.grid(axis='y',alpha=.25);ax.legend()
fig2.savefig(OUT/'phase6_violation_comparison.png',dpi=170)
plt.close(fig2)
print('\nSaved phase6_final_comparison.csv, phase6_capacity_ber.png, phase6_violation_comparison.png')
print('Note: 5% is a validation design criterion, not a 5G standard; no coding/fading/NR stack.')
