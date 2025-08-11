import { useCallback, useMemo, useState } from 'react';
import { View, StyleSheet } from 'react-native';
import { Chat } from '@flyerhq/react-native-chat-ui';
import type { MessageType } from '@flyerhq/react-native-chat-ui';
import { useAI } from '../context/AIContext';
import type { ChatMessage } from '../context/AIContext';

const convertFromLibraryMessage = (msg: MessageType.Text): ChatMessage => ({
  role: msg.author.id === 'user' ? 'user' : 'assistant',
  content: msg.text,
});

const user = { id: 'user' };
const assistant = { id: 'assistant' };

export default function ChatScreen() {
  const { runChat, isLoaded } = useAI();
  const [messages, setMessages] = useState<MessageType.Text[]>([]);
  const [isRunning, setIsRunning] = useState(false);

  const canSend = useMemo(() => isLoaded && !isRunning, [isLoaded, isRunning]);

  const handleSendPress = useCallback(
    async (message: MessageType.PartialText) => {
      if (!canSend) return;

      const userMessage: MessageType.Text = {
        author: user,
        id: Date.now().toString(),
        text: message.text,
        type: 'text',
      };

      const assistantMessage: MessageType.Text = {
        author: assistant,
        id: (Date.now() + 1).toString(),
        text: '',
        type: 'text',
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setIsRunning(true);

      try {
        const chatMessages: ChatMessage[] = [
          ...messages.map(convertFromLibraryMessage),
          convertFromLibraryMessage(userMessage),
        ];

        await runChat(chatMessages, (token) => {
          setMessages((prev) => {
            const next = [...prev];
            const lastMessage = next[next.length - 1];
            if (lastMessage && lastMessage.author.id === 'assistant') {
              lastMessage.text += token;
            }
            return next;
          });
        });
      } catch (e: any) {
        console.error(e);
        const errorMessage: MessageType.Text = {
          author: assistant,
          id: (Date.now() + 2).toString(),
          text: `Error: ${e?.message ?? e}`,
          type: 'text',
        };
        setMessages((prev) => [...prev, errorMessage]);
      } finally {
        setIsRunning(false);
      }
    },
    [canSend, messages, runChat]
  );

  return (
    <View style={styles.container}>
      <Chat
        messages={[...messages].reverse()}
        onSendPress={handleSendPress}
        user={user}
        textInputProps={{
          placeholder: 'Type a message...',
        }}
        showUserAvatars
        showUserNames
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});
