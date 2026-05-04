const chips = document.getElementById('chips');
const input = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const content = document.querySelector('.content');

chips.addEventListener('click', e => {
  const chip = e.target.closest('.chip');
  if(!chip) return;
  input.value = chip.textContent.trim();
  input.focus();
});

function appendUserMessage(text){
  const wrapper = document.createElement('div');
  wrapper.style.display='flex';
  wrapper.style.justifyContent='flex-end';
  wrapper.style.marginTop='14px';
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

function appendAssistantReply(text){
  const entry = document.createElement('div');
  entry.className = 'assistant-entry';
  entry.style.marginTop='12px';
  const icon = document.createElement('div');
  icon.className = 'assistant-icon';
  icon.textContent = '★';
  const msg = document.createElement('div');
  msg.className = 'message';
  msg.textContent = text;
  entry.appendChild(icon);
  entry.appendChild(msg);
  content.appendChild(entry);
  const ts = document.getElementById('time');
  const now = new Date();
  ts.textContent = now.getHours().toString().padStart(2,'0') + ':' + now.getMinutes().toString().padStart(2,'0');
  content.scrollTop = content.scrollHeight;
}

sendBtn.addEventListener('click', () => {
  const text = input.value.trim();
  if(!text) return;
  appendUserMessage(text);
  input.value = '';
  setTimeout(()=> appendAssistantReply("Thanks — I can help with: registration, navigation, exam timetables, results access and more. What would you like to do next?"), 700);
});

input.addEventListener('keydown', e => {
  if(e.key === 'Enter'){ e.preventDefault(); sendBtn.click(); }
});
