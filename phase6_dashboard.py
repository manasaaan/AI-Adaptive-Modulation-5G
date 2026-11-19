"""Phase 6 validated Streamlit dashboard.
Keep phase6_final_validation.py in the SAME folder.
Run: python -m streamlit run phase6_dashboard.py
"""
from pathlib import Path
import contextlib
import importlib.util
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title='AI Adaptive Modulation | Validated Phase 6', layout='wide')

@st.cache_resource(show_spinner='Loading Phase 6 validated experiment (training and bit simulation)...')
def load_validation():
    path = Path(__file__).resolve().parent / 'phase6_final_validation.py'
    if not path.exists():
        raise FileNotFoundError(f'Place phase6_final_validation.py beside this dashboard: {path}')
    spec = importlib.util.spec_from_file_location('phase6_validation_engine', path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module

try:
    v = load_validation()
except Exception as exc:
    st.error(f'Could not load Phase 6 validation: {exc}')
    st.stop()

models = {r['family']: r for r in v.selected_models}
original = models['Original']
tuned = models['Tuned RF']
gboost = models['Gradient Boosting']

def evaluate(pred, margin):
    choices = np.array([v.choose(float(p), margin) for p in pred], dtype=int)
    errors = np.zeros(len(choices), dtype=np.int64)
    sent = np.zeros(len(choices), dtype=np.int64)
    for j, M in enumerate(v.M_VALUES):
        mask = choices == M
        errors[mask] = v.errors_by_M[mask, j]
        sent[mask] = v.bits_by_M[mask, j]
    per_step = np.divide(errors, sent, out=np.full(len(choices), np.nan), where=sent > 0)
    tx = int(np.count_nonzero(choices))
    total_bits, total_errors = int(sent.sum()), int(errors.sum())
    violations = int(np.count_nonzero(per_step > v.TARGET_BER))
    analytical = int(sum(v.estimated_ber(int(M), float(db)) > v.TARGET_BER
                         for M, db in zip(choices[choices > 0], v.ytest[choices > 0])))
    return dict(choices=choices, errors=errors, sent=sent, step_ber=per_step,
                tx=tx, bits=total_bits, errors_total=total_errors,
                avg_bits=float(np.log2(np.maximum(choices, 1)).mean()),
                ber=total_errors/total_bits if total_bits else np.nan,
                violations=violations, analytical=analytical,
                violation_rate=violations/tx if tx else np.nan,
                correct_bits=(total_bits-total_errors)/(len(choices)*v.SYMBOLS_PER_STEP))

st.title('AI-Assisted Adaptive Modulation — Validated Phase 6')
st.caption('Phase 4C trained models + Phase 5A Gray-coded QAM bit-level AWGN | Synthetic channel, not a full 5G NR implementation')
with st.sidebar:
    st.header('Experiment controls')
    margin = st.slider('Tuned RF safety margin (dB)', 0.0, 6.0, float(tuned['margin']), 0.5)
    st.caption(f'Validated Phase 4C margin: {tuned["margin"]:.1f} dB. Other margins are exploratory and do not retrain AI.')
    show_steps = st.slider('Number of test steps displayed', 30, len(v.ytest), len(v.ytest))
    st.markdown('**Fixed validation settings**')
    st.write(f'Target BER: {v.TARGET_BER:g}')
    st.write(f'{v.SYMBOLS_PER_STEP:,} symbols per transmitting step')
    st.write(f'Train {len(v.ytrain)} / validation {len(v.yval)} / test {len(v.ytest)}')
    st.write(f'Channel seed: {v.CHANNEL_SEED}; bit/noise seed: {v.BITS_SEED}')
    st.info('Margin changes the rule-based modulation choice, not the trained Random Forest prediction.')

preds = {
    'Original RF — No margin': (original['pred_test'], 0.),
    'Original RF — Validation margin': (original['pred_test'], float(original['margin'])),
    'Tuned RF — No margin': (tuned['pred_test'], 0.),
    'Tuned RF — Validation margin': (tuned['pred_test'], float(tuned['margin'])),
    'Gradient Boosting — No margin': (gboost['pred_test'], 0.),
    'Gradient Boosting — Validation margin': (gboost['pred_test'], float(gboost['margin'])),
    'Last-value — no margin': (v.Xtest[:, -1], 0.),
    'Oracle — ideal': (v.ytest, 0.),
}
results = {name: evaluate(pred, m) for name, (pred, m) in preds.items()}
# This additional interactive result does not replace the validated, fixed-margin row.
interactive = evaluate(tuned['pred_test'], margin)

if margin != float(tuned['margin']):
    st.warning(f'Exploratory margin {margin:.1f} dB selected. Validated Phase 6 comparison below remains fixed at {tuned["margin"]:.1f} dB.')

c1,c2,c3,c4 = st.columns(4)
c1.metric('Tuned RF test MAE', f'{tuned["test_mae"]:.4f} dB')
c2.metric('Average raw bits/step', f'{interactive["avg_bits"]:.3f}')
c3.metric('Measured BER', f'{interactive["ber"]:.6f}')
c4.metric('BER-violating TX steps', f'{interactive["violations"]}/{interactive["tx"]}')

st.subheader('1. Channel quality and AI prediction')
steps = np.arange(show_steps)
fig, ax = plt.subplots(figsize=(12, 3.7))
ax.plot(steps, v.ytest[:show_steps], label='Actual Eb/N0', linewidth=1.5)
ax.plot(steps, tuned['pred_test'][:show_steps], label='Tuned RF prediction', linewidth=1.2)
ax.plot(steps, tuned['pred_test'][:show_steps]-margin, label='Prediction minus margin', linewidth=1.2)
ax.set(xlabel='Held-out test step', ylabel='Eb/N0 (dB)'); ax.grid(alpha=.25); ax.legend()
st.pyplot(fig); plt.close(fig)

st.subheader('2. Adaptive modulation decisions')
fig, ax = plt.subplots(figsize=(12, 3.3))
for label, choices in [('Original RF (0 dB)', results['Original RF — No margin']['choices']),
                       (f'Tuned RF ({margin:.1f} dB)', interactive['choices'])]:
    levels = np.array([{0:0,4:1,16:2,64:3}[int(M)] for M in choices[:show_steps]])
    ax.step(steps, levels, where='mid', label=label)
ax.set_yticks([0,1,2,3], ['No TX','QPSK','16-QAM','64-QAM'])
ax.set(xlabel='Held-out test step', ylabel='Modulation'); ax.grid(alpha=.25); ax.legend()
st.pyplot(fig); plt.close(fig)

st.subheader('3. Measured bit error rate per transmitting step')
fig, ax = plt.subplots(figsize=(12, 3.5))
for label, r in [('Original RF (0 dB)', results['Original RF — No margin']),
                 (f'Tuned RF ({margin:.1f} dB)', interactive)]:
    ax.plot(steps, r['step_ber'][:show_steps], marker='.', markersize=3, linewidth=.7, label=label)
ax.axhline(v.TARGET_BER, color='black', linestyle='--', label='BER target = 0.001')
ax.set_yscale('symlog', linthresh=1e-5)
ax.set(xlabel='Held-out test step', ylabel='Measured BER'); ax.grid(alpha=.25); ax.legend()
st.pyplot(fig); plt.close(fig)
st.caption('No TX has no measured BER. A measured zero means no errors observed in that finite transmission.')

st.subheader('4. Final validated controller comparison')
rows=[]
for name, r in results.items():
    rows.append({'Controller':name, 'Margin (dB)':preds[name][1], 'TX steps':r['tx'],
                 'Avg raw bits/step':round(r['avg_bits'],3), 'Measured BER':round(r['ber'],6),
                 'Measured violations':f'{r["violations"]}/{r["tx"]}',
                 'Violation rate (%)':round(100*r['violation_rate'],2),
                 'Analytical violations':f'{r["analytical"]}/{r["tx"]}',
                 'Correct bits/opportunity':round(r['correct_bits'],3)})
df = pd.DataFrame(rows)
st.dataframe(df, hide_index=True, use_container_width=True)
st.download_button('Download validated comparison CSV', df.to_csv(index=False),
                   file_name='phase6_dashboard_comparison.csv', mime='text/csv')
st.caption('These eight rows use the fixed Phase 4C validation-selected margins and reproduce Phase 6. Changing the sidebar margin does not alter this table.')

st.subheader('5. Inspect one transmission step')
step = st.slider('Select held-out test step', 0, len(v.ytest)-1, 0)
actual_db = float(v.ytest[step]); predicted_db = float(tuned['pred_test'][step]); adjusted_db = predicted_db-margin
M = int(interactive['choices'][step]); mod_name = {0:'No TX',4:'QPSK',16:'16-QAM',64:'64-QAM'}[M]
a,b,c = st.columns(3)
a.metric('Actual Eb/N0', f'{actual_db:.2f} dB')
b.metric('AI-predicted Eb/N0', f'{predicted_db:.2f} dB')
c.metric('Selected modulation', mod_name)

st.subheader('6. AI Decision Explanation')
st.caption('Random Forest predicts channel quality; a BER-threshold decision rule chooses the modulation. This is not a generative AI chatbot.')
a,b,c = st.columns(3)
a.metric('AI prediction', f'{predicted_db:.2f} dB')
b.metric('Safety adjustment', f'-{margin:.2f} dB')
c.metric('Adjusted Eb/N0', f'{adjusted_db:.2f} dB')
comparison=[]
for candidate in v.M_VALUES:
    estimate=v.estimated_ber(candidate,adjusted_db)
    comparison.append({'Modulation':{4:'QPSK',16:'16-QAM',64:'64-QAM'}[candidate],
                       'Bits/symbol':int(np.log2(candidate)),
                       'Estimated BER':f'{estimate:.6g}',
                       'Meets BER target?':'Yes' if estimate<=v.TARGET_BER else 'No'})
st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)
if M == 0:
    st.warning(f'At step {step}, the adjusted prediction is {adjusted_db:.2f} dB. No available modulation meets the estimated BER target of {v.TARGET_BER:g}; the controller selects No TX.')
else:
    errors=int(interactive['errors'][step]); total=int(interactive['sent'][step]); estimated=v.estimated_ber(M,adjusted_db)
    st.success(f'At step {step}, the tuned RF predicts {predicted_db:.2f} dB. After subtracting the {margin:.1f} dB margin, the controller selects {mod_name} ({int(np.log2(M))} bits/symbol), the highest-rate option meeting the estimated BER target ({estimated:.6g} ≤ {v.TARGET_BER:g}).')
    st.write(f'**Simulated outcome:** Actual Eb/N0 = {actual_db:.2f} dB. The receiver recovered {total-errors:,} of {total:,} transmitted bits correctly; {errors:,} bit errors, measured BER = {interactive["step_ber"][step]:.6g}. Actual Eb/N0 is used for channel simulation, not for the AI decision.')
st.caption('Change the margin or the selected test step to update the explanation and interactive graphs.')
st.info('Research limitations: synthetic time-varying Eb/N0 with conditional AWGN, no physical fading, channel coding, retransmission or 5G NR stack. The 5% step-violation criterion is an experimental design choice, not a 5G standard. Finite-bit measured BER can exceed analytical expectations.')
