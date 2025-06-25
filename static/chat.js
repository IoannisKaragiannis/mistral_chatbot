// static/chat.js
document.addEventListener("DOMContentLoaded", () => {
  const msgs    = document.getElementById("messages");
  const inp     = document.getElementById("prompt");
  const send    = document.getElementById("send");
  const newChat = document.getElementById("new-chat");
  let history   = [];

  newChat.addEventListener("click", () => {
    history = [];
    msgs.innerHTML = "";
    inp.value = "";
    inp.focus();
  });

  function render(role, text) {
    const div = document.createElement("div");
    div.classList.add("msg", role);

    const lbl = document.createElement("div");
    lbl.className = "label";
    lbl.textContent = role === "user" ? "You" : "Assistant";
    div.appendChild(lbl);

    const p = document.createElement("p");
    p.textContent = text;
    div.appendChild(p);

    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
  }

  async function clickSend() {
    const txt = inp.value.trim();
    if (!txt) return;
    history.push({ role: "user", content: txt });
    render("user", txt);
    inp.value = "";
    send.disabled = true;

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history })
      });
      if (!res.ok) throw new Error(res.statusText);
      const { role, content } = await res.json();
      history.push({ role, content });
      render("assistant", content);
    } catch (e) {
      render("assistant", `Error: ${e.message}`);
    } finally {
      send.disabled = false;
      inp.focus();
    }
  }

  send.addEventListener("click", clickSend);
  inp.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      clickSend();
    }
  });
});