import { useState, useEffect, type KeyboardEvent } from 'react';
import { ArrowUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

const TOP_K_OPTIONS = [5, 10, 15, 20, 25] as const;

interface ChatInputProps {
  onSendMessage: (payload: { message: string; topK: number }) => Promise<void> | void;
  isLoading?: boolean;
  topK: number;
  onTopKChange: (value: number) => void;
  prefillQuery?: string;
}

export function ChatInput({ onSendMessage, isLoading, topK, onTopKChange, prefillQuery }: ChatInputProps) {
  const [message, setMessage] = useState('');

  useEffect(() => {
    if (prefillQuery) {
      setMessage(prefillQuery);
    }
  }, [prefillQuery]);

  const handleSubmit = async () => {
    const trimmed = message.trim();
    if (isLoading) return;
    if (!trimmed) return;

    setMessage('');
    await onSendMessage({ message: trimmed, topK });
  };

  return (
    <div className="w-full max-w-3xl mx-auto px-4">
      <div className="flex items-center gap-2 mb-1 px-1">
        <label className="text-xs text-muted-foreground">Top resultados:</label>
        <select
          value={topK}
          onChange={(e) => onTopKChange(Number(e.target.value))}
          disabled={isLoading}
          className="text-xs bg-background border border-border rounded px-2 py-0.5 text-foreground outline-none focus:ring-1 focus:ring-ring"
        >
          {TOP_K_OPTIONS.map((k) => (
            <option key={k} value={k}>{k}</option>
          ))}
        </select>
      </div>
      <div className="chat-input-container flex items-center gap-2 p-3">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e: KeyboardEvent<HTMLInputElement>) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              void handleSubmit();
            }
          }}
          placeholder="Digite sua pergunta"
          className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground outline-none"
        />

        <Button
          onClick={handleSubmit}
          disabled={!message.trim() || isLoading}
          size="icon"
          className={cn(
            'h-9 w-9 rounded-full flex-shrink-0 transition-colors',
            message.trim()
              ? 'bg-foreground text-background hover:bg-foreground/90'
              : 'bg-muted text-muted-foreground'
          )}
        >
          <ArrowUp className="h-5 w-5" />
        </Button>
      </div>
    </div>
  );
}

