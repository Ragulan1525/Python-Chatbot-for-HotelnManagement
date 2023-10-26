const chatMessages = document.getElementById("chat-messages");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");

function appendMessage(sender, message) {
	const messageDiv = document.createElement("div");
	messageDiv.classList.add("mb-2", "p-2", "rounded", "max-w-fit", "ml-auto");
	if (sender === "user") {
		messageDiv.innerHTML = `<div class="bg-green-500 text-white px-4 py-2 rounded-md">${message}</div>`;
	} else {
		messageDiv.classList.toggle("ml-auto");
		messageDiv.innerHTML = `<div class="bg-gray-300 px-4 py-2 rounded-md">${message}</div>`;
	}
	chatMessages.appendChild(messageDiv);
	chatMessages.scrollTop = chatMessages.scrollHeight;
}

function sendMessage() {
	const message = userInput.value;
	if (message.trim() === "") return;
	appendMessage("user", message);
	userInput.value = "";
	// Simulate response from the bot (in this example, a simple echo)
	setTimeout(() => {
		appendMessage("bot", message);
	}, 500);
}

sendButton.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", (e) => {
	if (e.key === "Enter") {
		sendMessage();
	}
});
