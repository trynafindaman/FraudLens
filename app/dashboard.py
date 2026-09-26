"""HTML Dashboard template for Real-Time FraudLens observability."""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>FraudLens — Real-Time Streaming Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0D1117;
      --card-bg: #161B22;
      --border: #30363D;
      --text: #C9D1D9;
      --text-dim: #8B949E;
      --red: #F85149;
      --green: #3FB950;
      --blue: #58A6FF;
      --mono: 'JetBrains Mono', monospace;
      --sans: 'Inter', sans-serif;
    }
    body {
      margin: 0; padding: 24px; background: var(--bg); color: var(--text);
      font-family: var(--sans); line-height: 1.5;
    }
    .header {
      display: flex; align-items: baseline; justify-content: space-between;
      border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px;
    }
    .header h1 { margin: 0; font-size: 24px; font-weight: 700; color: #FFF; }
    .badge {
      font-family: var(--mono); font-size: 12px; padding: 4px 10px;
      border-radius: 12px; background: rgba(63, 185, 80, 0.15); color: var(--green); border: 1px solid var(--green);
    }
    .kpi-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 28px;
    }
    .kpi-card {
      background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 18px;
    }
    .kpi-label { font-size: 13px; color: var(--text-dim); text-transform: uppercase; font-family: var(--mono); }
    .kpi-val { font-size: 32px; font-weight: 700; margin-top: 6px; font-family: var(--mono); color: #FFF; }
    .kpi-val.danger { color: var(--red); }
    .table-card {
      background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px;
    }
    .table-title { margin: 0 0 16px; font-size: 16px; font-weight: 600; color: #FFF; }
    table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 13px; }
    th { text-align: left; padding: 10px 12px; color: var(--text-dim); border-bottom: 1px solid var(--border); font-size: 11px; text-transform: uppercase; }
    td { padding: 10px 12px; border-bottom: 1px solid var(--border); }
    tr:last-child td { border-bottom: none; }
    .tag { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    .tag.normal { background: rgba(63, 185, 80, 0.15); color: var(--green); }
    .tag.suspicious { background: rgba(248, 81, 73, 0.2); color: var(--red); border: 1px solid var(--red); }
  </style>
</head>
<body>
  <div class="header">
    <h1>🛡️ FraudLens Real-Time Detection API</h1>
    <span class="badge">● LIVE STREAMING</span>
  </div>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Total Transactions Scored</div>
      <div class="kpi-val" id="totalRequests">0</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Fraudulent / Suspicious</div>
      <div class="kpi-val danger" id="suspiciousCount">0</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Current Flag Rate</div>
      <div class="kpi-val" id="flagRate">0.0%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Average Scoring Latency</div>
      <div class="kpi-val" id="avgLatency">0.0 ms</div>
    </div>
  </div>

  <div class="table-card">
    <div class="table-title">Recent Real-Time Transaction Stream</div>
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Tx ID</th>
          <th>Amount ($)</th>
          <th>Risk Score</th>
          <th>Status</th>
          <th>Latency</th>
        </tr>
      </thead>
      <tbody id="txBody">
        <tr><td colspan="6" style="text-align:center; color: var(--text-dim);">Waiting for stream traffic...</td></tr>
      </tbody>
    </table>
  </div>

  <script>
    async function updateDashboard() {
      try {
        const res = await fetch('/dashboard/data');
        const data = await res.json();

        document.getElementById('totalRequests').innerText = data.total_requests.toLocaleString();
        document.getElementById('suspiciousCount').innerText = data.suspicious_count.toLocaleString();
        document.getElementById('flagRate').innerText = data.flag_rate_pct.toFixed(2) + '%';
        document.getElementById('avgLatency').innerText = data.avg_latency_ms.toFixed(2) + ' ms';

        const tbody = document.getElementById('txBody');
        if (data.recent_transactions && data.recent_transactions.length > 0) {
          tbody.innerHTML = data.recent_transactions.map(tx => {
            const statusClass = tx.is_suspicious ? 'suspicious' : 'normal';
            const statusLabel = tx.is_suspicious ? 'FLAGGED' : 'LEGIT';
            return `<tr>
              <td>${tx.time}</td>
              <td><code>${tx.transaction_id}</code></td>
              <td>$${tx.amount.toFixed(2)}</td>
              <td><strong>${tx.risk_score.toFixed(4)}</strong></td>
              <td><span class="tag ${statusClass}">${statusLabel}</span></td>
              <td>${tx.latency_ms.toFixed(1)} ms</td>
            </tr>`;
          }).join('');
        }
      } catch (err) {
        console.error("Dashboard poll failed:", err);
      }
    }
    setInterval(updateDashboard, 1500);
    updateDashboard();
  </script>
</body>
</html>"""
