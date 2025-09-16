async function tuneThenTrain() {
  const statusEl = document.getElementById('status');
  const trials = parseInt(document.getElementById('trials').value, 10) || 10;
  const dataset = document.getElementById('dataset').value;
  statusEl.textContent = `Running tuning (${trials} trials)…`;
  try {
    const tuneRes = await fetch('http://127.0.0.1:8000/tune', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset, trials })
    });
    const tune = await tuneRes.json();
    if (tune.error) throw new Error(tune.error);
    document.getElementById('bestParams').textContent = JSON.stringify(tune, null, 2);

    // apply best params into form (where relevant)
    const p = tune.best_params || {};
    if (p.n_lags) document.getElementById('n_lags').value = p.n_lags;
    if (p.epochs) document.getElementById('epochs').value = p.epochs;
    if (p.learning_rate) document.getElementById('learning_rate').value = p.learning_rate;
    if (p.batch_size) document.getElementById('batch_size').value = p.batch_size;
    if (p.number_nodes) document.getElementById('number_nodes').value = p.number_nodes;

    // now run training with applied params
    await run();
  } catch (e) {
    statusEl.textContent = 'Error: ' + e.message;
  }
}
