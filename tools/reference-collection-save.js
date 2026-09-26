// Shared, user-initiated JSON persistence for the detector and layout editors.
// A successful file-picker write is confirmed; a browser download is only requested.
window.HTH_REFERENCE_SAVE = (() => {
  function createJsonSaver() {
    let handle = null;
    let handleName = null;

    async function save(name, contents) {
      if (typeof name !== 'string' || !name.endsWith('.json')) throw Error('A JSON filename is required.');
      if (typeof contents !== 'string') throw Error('JSON contents must be text.');
      if (typeof window.showSaveFilePicker === 'function') {
        if (handleName !== name) {
          handle = null;
          handleName = name;
        }
        if (!handle) {
          handle = await window.showSaveFilePicker({
            suggestedName: name,
            types: [{description: 'JSON document', accept: {'application/json': ['.json']}}],
          });
        }
        let permission = await handle.queryPermission({mode: 'readwrite'});
        if (permission !== 'granted') permission = await handle.requestPermission({mode: 'readwrite'});
        if (permission !== 'granted') throw Error('Write permission was not granted for the selected JSON file.');
        const writable = await handle.createWritable();
        try {
          await writable.write(contents);
        } finally {
          await writable.close();
        }
        return {kind: 'written', filename: handle.name || name};
      }

      const url = URL.createObjectURL(new Blob([contents], {type: 'application/json'}));
      const link = document.createElement('a');
      link.href = url;
      link.download = name;
      link.style.display = 'none';
      document.body.appendChild(link);
      link.click();
      setTimeout(() => {URL.revokeObjectURL(url); link.remove();}, 60000);
      return {kind: 'download-requested', filename: name};
    }

    return {save};
  }

  return {createJsonSaver};
})();
