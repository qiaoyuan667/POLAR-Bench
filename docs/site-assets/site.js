// Progressive enhancement: all content and navigation work without JavaScript.
for (const button of document.querySelectorAll('[data-copy]')) {
  if (!navigator.clipboard || !window.isSecureContext) continue;
  button.hidden = false;
  button.addEventListener('click', async () => {
    const source = document.getElementById(button.dataset.copy);
    const status = document.getElementById('copy-status');
    if (!source) return;
    try {
      await navigator.clipboard.writeText(source.textContent);
      button.textContent = 'Copied';
      if (status) status.textContent = 'Copied to clipboard.';
      window.setTimeout(() => { button.textContent = 'Copy'; }, 2000);
    } catch {
      button.textContent = 'Select text to copy';
      if (status) status.textContent = 'Clipboard access was unavailable. Select and copy the code directly.';
    }
  });
}
