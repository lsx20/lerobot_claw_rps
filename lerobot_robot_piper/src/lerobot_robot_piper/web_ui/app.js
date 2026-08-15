const pages = [...document.querySelectorAll('.page')];
const errorText = document.querySelector('#error');
const tactileSummary = document.querySelector('#tactile-summary');
const disableButton = document.querySelector('#disable-button');
let tactileTimer = null;
let statusTimer = null;

const stopTactile = () => {
  if (tactileTimer !== null) {
    clearInterval(tactileTimer);
    tactileTimer = null;
  }
};

const show = name => {
  if (name !== 'tactile') {
    stopTactile();
  }
  pages.forEach(page => page.classList.toggle('active', page.dataset.page === name));
};

const request = async url => {
  const response = await fetch(url, {method: 'POST'});
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || '请求失败');
  }
  return data;
};

const getJson = async url => {
  const response = await fetch(url, {cache: 'no-store'});
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || '请求失败');
  }
  return data;
};

const postJson = async (url, payload) => {
  const response = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || '请求失败');
  }
  return data;
};

const setError = error => {
  errorText.textContent = error.message || String(error);
  show('error');
};

const runOnce = async (button, task) => {
  if (button.disabled) return;
  button.disabled = true;
  try {
    await task();
  } catch (error) {
    setError(error);
  } finally {
    button.disabled = false;
  }
};

const stopStatusPolling = () => {
  if (statusTimer !== null) {
    clearInterval(statusTimer);
    statusTimer = null;
  }
};

const pollRemoteStatus = async () => {
  const data = await getJson('/api/status');
  if (data.stage === 'remote_done') {
    stopStatusPolling();
    show('remote-success');
  } else if (data.stage === 'remote_failed') {
    stopStatusPolling();
    show('remote-fail');
  } else if (data.stage === 'remote_running') {
    show('remote-waiting');
  }
};

const startRemoteStatusPolling = () => {
  stopStatusPolling();
  statusTimer = setInterval(() => pollRemoteStatus().catch(setError), 1000);
};

document.querySelectorAll('[data-action="home"]').forEach(button => {
  button.onclick = () => {
    stopStatusPolling();
    show('home');
  };
});

disableButton.onclick = async () => {
  const password = window.prompt('请输入调试密码');
  if (!password) return;
  disableButton.disabled = true;
  try {
    await postJson('/api/disable', {password});
    stopStatusPolling();
    stopTactile();
    show('home');
  } catch (error) {
    setError(error);
  } finally {
    disableButton.disabled = false;
  }
};

document.querySelector('[data-action="rps-mode"]').onclick = async event => {
  await runOnce(event.currentTarget, async () => {
    stopStatusPolling();
    await request('/api/mode/rps');
    show('instruction');
  });
};

document.querySelector('[data-action="recognize"]').onclick = async event => {
  await runOnce(event.currentTarget, async () => {
    show('waiting');
    const data = await request('/api/game/start');
    show(data.result === 'PLAYER_WIN' ? 'win' : 'retry');
  });
};

document.querySelector('[data-action="retry"]').onclick = () => show('instruction');

document.querySelector('[data-action="lottery"]').onclick = async event => {
  await runOnce(event.currentTarget, async () => {
    await request('/api/lottery/start');
    show('tactile');
    startTactile();
  });
};

const refreshTactile = async () => {
  const data = await getJson('/api/tactile/summary');
  tactileSummary.innerHTML = data.html;
};

const startTactile = () => {
  stopTactile();
  tactileSummary.innerHTML = '<p>正在读取触觉数据...</p>';
  refreshTactile().catch(setError);
  tactileTimer = setInterval(() => refreshTactile().catch(setError), 500);
};

document.querySelector('[data-action="remote-mode"]').onclick = async event => {
  await runOnce(event.currentTarget, async () => {
    show('remote');
    await request('/api/mode/remote');
  });
};

const runRemote = async event => {
  await runOnce(event.currentTarget, async () => {
    show('remote-waiting');
    const data = await request('/api/remote/run');
    if (data.result === 'SUCCESS') {
      show('remote-success');
    } else if (data.result === 'FAILED') {
      show('remote-fail');
    } else {
      startRemoteStatusPolling();
    }
  });
};

document.querySelector('[data-action="remote-run"]').onclick = runRemote;
document.querySelector('[data-action="remote-retry"]').onclick = runRemote;
