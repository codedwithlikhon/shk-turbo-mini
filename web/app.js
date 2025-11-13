// Minimal client-side viewer: reads logs and metrics from /api endpoints (server not included)
async function fetchJSON(path){ try{let r=await fetch(path); if(!r.ok) return null; return await r.json(); }catch(e){return null} }
async function refresh(){
  const logs = await fetchJSON('/api/logs') || {text:"No logs"};
  document.getElementById('logs').textContent = logs.text;
  const metrics = await fetchJSON('/api/metrics') || {items:[]};
  document.getElementById('metrics').innerHTML = metrics.items.map(k=>`<div>${k.name}: ${k.value}</div>`).join('');
}
document.getElementById('refresh').addEventListener('click', refresh);
window.onload = refresh;