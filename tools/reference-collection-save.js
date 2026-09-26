// Shared, user-initiated JSON persistence for the detector and layout editors.
// A successful file-picker write is confirmed; a browser download is only requested.
window.HTH_REFERENCE_SAVE = (() => {
  function createJsonSaver() {
    let handle = null;
    let handleName = null;

    async function save(name, contents, {saveAs = false} = {}) {
      if (typeof name !== 'string' || !name.endsWith('.json')) throw Error('A JSON filename is required.');
      if (typeof contents !== 'string') throw Error('JSON contents must be text.');
      if (typeof window.showSaveFilePicker === 'function') {
        let destination = handleName === name ? handle : null;
        if (saveAs || !destination) {
          destination = await window.showSaveFilePicker({
            suggestedName: saveAs && handle?.name ? handle.name : name,
            types: [{description: 'JSON document', accept: {'application/json': ['.json']}}],
          });
        }
        let permission = await destination.queryPermission({mode: 'readwrite'});
        if (permission !== 'granted') permission = await destination.requestPermission({mode: 'readwrite'});
        if (permission !== 'granted') throw Error('Write permission was not granted for the selected JSON file.');
        const writable = await destination.createWritable();
        try {
          await writable.write(contents);
        } finally {
          await writable.close();
        }
        handle = destination;
        handleName = name;
        return {kind: 'written', filename: handle.name || name};
      }

      if (saveAs) throw Error('This browser cannot choose a new save location. Enable “Ask where to save each file” in its download settings.');

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
