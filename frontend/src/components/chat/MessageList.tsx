import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Message } from '@/lib/api';
import { cn } from '@/lib/utils';
import { Bot, User } from 'lucide-react';
import { useEffect, useRef } from 'react';

interface MessageListProps {
  messages: Message[];
  isLoading?: boolean;
}

export function MessageList({ messages, isLoading }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  if (messages.length === 0 && !isLoading) {
    return null;
  }

  return (
    <div className="w-full">
      <div className="py-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={cn(
              'message-bubble fade-in',
              message.role === 'user' ? 'bg-transparent' : 'bg-message-assistant'
            )}
          >
            <div className="flex gap-4">
              <Avatar className="h-8 w-8 flex-shrink-0">
                <AvatarFallback
                  className={cn(
                    'text-xs',
                    message.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-success text-success-foreground'
                  )}
                >
                  {message.role === 'user' ? (
                    <User className="h-4 w-4" />
                  ) : (
                    <Bot className="h-4 w-4" />
                  )}
                </AvatarFallback>
              </Avatar>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium mb-1">
                  {message.role === 'user' ? 'Você' : 'Semantic Search'}
                </p>
                <div className="text-sm text-foreground/90 whitespace-pre-wrap break-words">
                  {message.content}
                </div>
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="message-bubble bg-message-assistant fade-in">
            <div className="flex gap-4">
              <Avatar className="h-8 w-8 flex-shrink-0">
                <AvatarFallback className="bg-success text-success-foreground text-xs">
                  <Bot className="h-4 w-4" />
                </AvatarFallback>
              </Avatar>
              <div className="flex-1">
                <p className="text-sm font-medium mb-2">Semantic Search</p>
                <div className="typing-indicator">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
