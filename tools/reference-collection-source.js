// Same-origin client for the small local release proxy. Each editor can load
// independently; the director can load once and share a verified result.
window.HTH_REFERENCE_SOURCE = (() => {
  'use strict';
  const launcher = 'python tools/reference-collection-director.py';
  const requireServer = () => {
    if (location.protocol === 'file:')
      throw Error(`Automatic release loading requires the local launcher: ${launcher}`);
  };
  const params = (sourceRepo, goldenSetId) => new URLSearchParams({ source_repo: sourceRepo.trim(), golden_set_id: goldenSetId.trim() });
  function resultsRepository(sourceRepo) {
    const match = sourceRepo.trim().match(/^https:\/\/github\.com\/([A-Za-z0-9_.-]+)\/([A-Za-z0-9_.-]+)\/?$/i);
    return match ? `https://github.com/${match[1]}/${match[2]}-results` : '';
  }
  async function readJson(response) {
    const payload = await response.json();
    if (!response.ok) throw Error(payload.error || `Release service returned HTTP ${response.status}`);
    return payload;
  }
  async function listGoldenSets(sourceRepo) {
    requireServer();
    const query = new URLSearchParams({ source_repo: sourceRepo.trim() });
    return (await readJson(await fetch(`/api/golden-sets?${query}`))).golden_sets;
  }
  async function loadRelease(sourceRepo, goldenSetId, status = () => {}) {
    requireServer();
    const query = params(sourceRepo, goldenSetId);
    status('Resolving frozen Golden Set release…');
    const release = await readJson(await fetch(`/api/reference-release?${query}`));
    status(`Verified ${release.tag} identity; downloading ${release.bundle.asset}…`);
    const response = await fetch(`/api/image-bundle?${query}`);
    if (!response.ok) throw Error((await response.json()).error || `Image download returned HTTP ${response.status}`);
    const total = release.bundle.size, chunks = [], reader = response.body.getReader();
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      status(`Downloading ${release.bundle.asset}: ${(received / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MiB`);
      if (received > total) throw Error('Image download exceeded the frozen bundle size.');
    }
    if (received !== total) throw Error(`Image download incomplete: ${received}/${total} bytes.`);
    const file = new File(chunks, release.bundle.asset, { type: 'application/zip' });
    const files = await window.HTH_REFERENCE_IMAGES.readBundle(file, release.bundle, release.golden_set, status);
    status(`${files.length} images verified for ${release.tag}.`);
    return { ...release, sourceRepo, files };
  }
  async function updateResults(sourceRepo, status = () => {}) {
    requireServer();
    status(`Checking ${resultsRepository(sourceRepo)} and running git pull --ff-only…`);
    const query = new URLSearchParams({ source_repo: sourceRepo.trim() });
    const result = await readJson(await fetch(`/api/results-update?${query}`, { method: 'POST' }));
    status(`Results checkout ${result.status}: ${result.after.slice(0, 12)}. Reopen the workspace picker to read updated files.`);
    return result;
  }
  return { listGoldenSets, loadRelease, resultsRepository, updateResults };
})();
