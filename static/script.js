const chips = document.getElementById('chips');
const input = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const content = document.querySelector('.content');
const documentUpload = document.getElementById('documentUpload');

chips.addEventListener('click', e => {
  const chip = e.target.closest('.chip');
  if (!chip) return;
  input.value = chip.textContent.trim();
  input.focus();
});

function appendUserMessage(text) {
  const wrapper = document.createElement('div');
  wrapper.style.display = 'flex';
  wrapper.style.justifyContent = 'flex-end';
  wrapper.style.marginTop = '14px';
  const bubble = document.createElement('div');
  bubble.textContent = text;
  bubble.style.background = 'linear-gradient(180deg,#e6f0ff,#dbeaff)';
  bubble.style.padding = '14px 16px';
  bubble.style.borderRadius = '14px';
  bubble.style.maxWidth = '82%';
  bubble.style.fontWeight = '600';
  wrapper.appendChild(bubble);
  content.appendChild(wrapper);
  content.scrollTop = content.scrollHeight;
}

function appendAssistantReply(text) {
  const entry = document.createElement('div');
  entry.className = 'assistant-entry';
  entry.style.marginTop = '12px';
  const icon = document.createElement('div');
  icon.className = 'assistant-icon';
  icon.textContent = '*';
  const msg = document.createElement('div');
  msg.className = 'message';
  msg.textContent = text;
  entry.appendChild(icon);
  entry.appendChild(msg);
  content.appendChild(entry);
  const ts = document.getElementById('time');
  const now = new Date();
  ts.textContent = now.getHours().toString().padStart(2, '0') + ':' + now.getMinutes().toString().padStart(2, '0');
  content.scrollTop = content.scrollHeight;
  return entry;
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;

  appendUserMessage(text);
  input.value = '';
  sendBtn.disabled = true;
  const thinkingEntry = appendAssistantReply('Thinking...');

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await response.json();

    if (!response.ok) {
      thinkingEntry.querySelector('.message').textContent = data.error || 'Something went wrong while processing your message.';
      return;
    }

    thinkingEntry.querySelector('.message').textContent = data.response;
  } catch (error) {
    thinkingEntry.querySelector('.message').textContent = 'Unable to reach the chatbot service. Please try again.';
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

sendBtn.addEventListener('click', sendMessage);

documentUpload.addEventListener('change', async () => {
  const files = Array.from(documentUpload.files || []);
  if (!files.length) return;

  const uploadEntry = appendAssistantReply(`Uploading ${files.length} document${files.length === 1 ? '' : 's'}...`);
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));
  documentUpload.disabled = true;

  try {
    const response = await fetch('/documents/upload', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();
    const loadedCount = (data.loaded_documents || []).length;
    const failedCount = (data.failed_documents || []).length;

    if (!response.ok && !loadedCount) {
      const failureDetails = (data.failed_documents || [])
        .map(item => `${item.filename}: ${item.error}`)
        .join('\n');
      uploadEntry.querySelector('.message').textContent =
        failureDetails || data.error || 'The document upload failed.';
      return;
    }

    const loadedDetails = (data.loaded_documents || [])
      .map(item => `${item.filename}: ${item.chunks_added} chunk${item.chunks_added === 1 ? '' : 's'}`)
      .join('\n');
    const failedDetails = (data.failed_documents || [])
      .map(item => `${item.filename}: ${item.error}`)
      .join('\n');

    uploadEntry.querySelector('.message').textContent = [
      `Uploaded ${loadedCount} document${loadedCount === 1 ? '' : 's'} into searchable chunks.`,
      loadedDetails,
      failedCount ? `${failedCount} failed:\n${failedDetails}` : ''
    ].filter(Boolean).join('\n\n');
  } catch (error) {
    uploadEntry.querySelector('.message').textContent = 'Unable to upload the selected documents.';
  } finally {
    documentUpload.disabled = false;
    documentUpload.value = '';
    input.focus();
  }
});

input.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    e.preventDefault();
    sendBtn.click();
  }
});
