#!/usr/bin/env node
'use strict';

// Dependency-free smoke test for the static browser workbench. It exercises
// the real app.js startup path with a small DOM/fetch shim (not a scoring
// reimplementation) so asynchronous initialization regressions are caught.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const appSource = fs.readFileSync(path.join(root, 'web', 'app.js'), 'utf8');
const contract = JSON.parse(fs.readFileSync(path.join(root, 'web', 'scoring_contract.json'), 'utf8'));

const defaults = {
  preset: 'mcm_mix',
  nComp: '4',
  nConc: '6',
  seed: '42',
  w_fpd: '0.30',
  w_rate: '0.15',
  w_amp: '0.15',
  w_stv: '0.25',
  w_tri: '0.15',
  th_low: '0.30',
  th_mod: '0.60',
};

class FakeElement {
  constructor(id) {
    this.id = id;
    this.value = defaults[id] ?? '';
    this.checked = id === 'includeToxic';
    this.style = {};
    this.innerHTML = '';
    this.textContent = '';
    this.children = [];
    this.listeners = {};
    this.files = [];
  }
  addEventListener(event, callback) { this.listeners[event] = callback; }
  appendChild(child) { this.children.push(child); return child; }
  remove() {}
  click() { if (typeof this.onclick === 'function') this.onclick(); }
}

async function runScenario({ failContract = false } = {}) {
  const elements = new Map();
  const alerts = [];
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new FakeElement(id));
      return elements.get(id);
    },
    createElement(tag) { return new FakeElement(tag); },
    body: { appendChild() {} },
  };
  const fetch = () => failContract
    ? Promise.reject(new Error('simulated fetch failure'))
    : Promise.resolve({ ok: true, json: () => Promise.resolve(contract) });

  vm.runInNewContext(appSource, { document, fetch, alert: (message) => alerts.push(String(message)) }, {
    filename: 'web/app.js',
    timeout: 2000,
  });
  await new Promise((resolve) => setImmediate(resolve));
  return { elements, alerts };
}

(async () => {
  const success = await runScenario();
  assert.equal(success.alerts.length, 0, 'startup must not show a premature contract-loading alert');
  assert.match(success.elements.get('out').innerHTML, /Summary — MCM \/ biodefense prioritization/,
    'successful contract load should initialize and render the demo');
  assert.match(success.elements.get('out').innerHTML, /CardioScore/,
    'rendered demo should contain score readings');
  assert.ok(success.elements.get('qcList').children.length > 0,
    'the demo should surface its quality-control log');

  const failure = await runScenario({ failContract: true });
  assert.match(failure.elements.get('out').textContent, /could not load its scoring contract/i,
    'contract-fetch failure should give the user an actionable on-page message');
  assert.equal(failure.alerts.length, 0, 'contract-fetch failure should not create an unhandled alert loop');

  process.stdout.write('Browser workbench smoke test passed (startup, scoring render, QC log, fetch failure).\n');
})().catch((error) => {
  process.stderr.write(String(error.stack || error) + '\n');
  process.exitCode = 1;
});
