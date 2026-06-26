const chips = document.getElementById('chips');
const input = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const content = document.querySelector('.content');

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

function renderHelpfulLinks(messageElement, links) {
  const existingLinks = messageElement.querySelector('.helpful-links');
  if (existingLinks) existingLinks.remove();

  if (!Array.isArray(links) || links.length === 0) return;

  const list = document.createElement('div');
  list.className = 'helpful-links';

  links.forEach(link => {
    if (!link || !link.url) return;
    const anchor = document.createElement('a');
    anchor.className = 'helpful-link';
    anchor.href = link.url;
    anchor.target = '_blank';
    anchor.rel = 'noopener noreferrer';
    anchor.textContent = link.title || 'Open link';
    list.appendChild(anchor);
  });

  if (list.children.length > 0) {
    messageElement.appendChild(list);
  }
}

function setAssistantReply(entry, text, links = []) {
  const messageElement = entry.querySelector('.message');
  messageElement.textContent = text;
  renderHelpfulLinks(messageElement, links);
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
      setAssistantReply(thinkingEntry, data.error || 'Something went wrong while processing your message.');
      return;
    }

    setAssistantReply(thinkingEntry, data.response, data.context?.helpful_links);
  } catch (error) {
    setAssistantReply(thinkingEntry, 'Unable to reach the chatbot service. Please try again.');
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

sendBtn.addEventListener('click', sendMessage);

input.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    e.preventDefault();
    sendBtn.click();
  }
});
