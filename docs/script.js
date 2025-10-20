// static/script.js
// Sends JSON to /predict, handles response, and fetches /history to show recent rows.

async function fetchHistoryAndRender() {
  try {
    const res = await fetch('/history');
    if (!res.ok) {
      const t = await res.text();
      document.getElementById('history').innerHTML = `<div class="error-box"><b>History load failed:</b> ${t}</div>`;
      return;
    }
    const json = await res.json();
    const rows = json.rows || [];

    if (rows.length === 0) {
      document.getElementById('history').innerHTML = `<p>No history yet.</p>`;
      return;
    }

    // Build a simple table
    let html = `<table id="history-table"><thead><tr><th>#</th><th>Time</th><th>Age</th><th>Result</th><th>Prob</th></tr></thead><tbody>`;
    rows.forEach((r, i) => {
      const t = new Date(r.created_at).toLocaleString();
      const prob = r.prob_high !== null && r.prob_high !== undefined ? (Number(r.prob_high).toFixed(2)) : 'N/A';
      html += `<tr>
        <td>${i+1}</td>
        <td>${t}</td>
        <td>${r.age ?? '-'}</td>
        <td style="color:${r.prediction === 'High Risk' ? '#d32f2f' : '#2e7d32'}">${r.prediction}</td>
        <td>${prob}</td>
      </tr>`;
    });
    html += `</tbody></table>`;
    document.getElementById('history').innerHTML = html;
  } catch (err) {
    document.getElementById('history').innerHTML = `<div class="error-box"><b>Unexpected:</b> ${err.message}</div>`;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predictForm');
  const resultEl = document.getElementById('result');
  const clearBtn = document.getElementById('clearBtn');

  // initial history load
  fetchHistoryAndRender();

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    resultEl.innerHTML = ''; // clear previous

    // collect values from form (works with name attributes)
    const fd = new FormData(form);
    const payload = {};
    fd.forEach((v, k) => { payload[k] = v; });

    // client-side basic validation for required keys
    const required = ['age','bilirubin','albumin','inr','creatinine'];
    for (const r of required) {
      if (!payload[r] || payload[r] === '') {
        resultEl.innerHTML = `<div class="error-box"><b>Error:</b> Missing required field: ${r}</div>`;
        return;
      }
    }

    try {
      const resp = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      // If server returned non-JSON (error page), read as text and display
      const contentType = resp.headers.get('content-type') || '';
      if (!resp.ok) {
        if (contentType.includes('application/json')) {
          const errJson = await resp.json();
          resultEl.innerHTML = `<div class="error-box"><b>Error:</b> ${errJson.error || JSON.stringify(errJson)}</div>`;
        } else {
          const txt = await resp.text();
          resultEl.innerHTML = `<div class="error-box"><b>Error:</b> ${txt}</div>`;
        }
        return;
      }

      // success — parse json
      const data = await resp.json();
      const probText = (data.probability_high_risk !== undefined && data.probability_high_risk !== null)
                        ? (Number(data.probability_high_risk).toFixed(3))
                        : 'N/A';

      resultEl.innerHTML = `
        <div class="result-box">
          <h3>✅ Prediction Result</h3>
          <p><b>Prediction:</b> ${data.prediction}</p>
          <p><b>Probability High Risk:</b> ${probText}</p>
          <p><b>Child-Pugh:</b> ${data.child_pugh}</p>
          <p><b>MELD Score:</b> ${data.meld_score ?? 'N/A'}</p>
        </div>
      `;
      // refresh history to show saved row
      fetchHistoryAndRender();
    } catch (err) {
      resultEl.innerHTML = `<div class="error-box"><b>Unexpected Error:</b> ${err.message}</div>`;
    }
  });

  // Reset button clears result and keeps form cleared
  clearBtn.addEventListener('click', () => {
    document.getElementById('result').innerHTML = '';
  });
});
