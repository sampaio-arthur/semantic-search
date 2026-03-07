import { Plus, MessageSquare, ChevronLeft, LogOut, Trash2, ClipboardList } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Conversation } from '@/lib/api';
import { useAuth } from '@/contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { cn } from '@/lib/utils';

interface ChatSidebarProps {
  conversations: Conversation[];
  activeConversationId: number | null;
  onSelectConversation: (id: number) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: number) => void;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

export function ChatSidebar({
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  isCollapsed,
  onToggleCollapse,
}: ChatSidebarProps) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const getUserInitials = (email: string) => {
    return email.substring(0, 2).toUpperCase();
  };

  return (
    <aside
      className={cn(
        'h-screen bg-sidebar flex flex-col border-r border-sidebar-border transition-all duration-300',
        isCollapsed ? 'w-0 overflow-hidden' : 'w-64'
      )}
    >
      <div className='p-3 flex items-center justify-between'>
        <Button
          variant='ghost'
          size='icon'
          onClick={onToggleCollapse}
          className='text-sidebar-foreground hover:bg-sidebar-accent'
        >
          <ChevronLeft className='h-5 w-5' />
        </Button>
        <Button
          variant='ghost'
          size='icon'
          onClick={onNewConversation}
          aria-label='Novo chat'
          className='text-sidebar-foreground hover:bg-sidebar-accent'
        >
          <Plus className='h-5 w-5' />
        </Button>
      </div>

      <div className='px-3 mb-2 space-y-1'>
        <Button
          onClick={onNewConversation}
          variant='ghost'
          className='w-full justify-start gap-3 text-sidebar-foreground hover:bg-sidebar-accent'
        >
          <Plus className='h-4 w-4' />
          Novo chat
        </Button>
        <Button
          onClick={() => navigate('/evaluation-queries')}
          variant='ghost'
          className='w-full justify-start gap-3 text-sidebar-foreground hover:bg-sidebar-accent'
        >
          <ClipboardList className='h-4 w-4' />
          Queries de Avaliação
        </Button>
      </div>

      <ScrollArea className='flex-1 px-2'>
        <div className='space-y-1'>
          <p className='px-3 py-2 text-xs text-muted-foreground font-medium'>
            Seus chats
          </p>
          {conversations.map((conversation) => {
            const title = conversation.title || 'Sem titulo';
            return (
              <div
                key={conversation.id}
                onClick={() => onSelectConversation(conversation.id)}
                role='button'
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    onSelectConversation(conversation.id);
                  }
                }}
                className={cn(
                  'sidebar-item grid grid-cols-[auto,1fr,auto] items-center w-full text-left text-sm text-sidebar-foreground min-w-0',
                  activeConversationId === conversation.id && 'active'
                )}
              >
                <MessageSquare className='h-4 w-4 flex-shrink-0' />
                <span
                  className='flex-1 min-w-0 truncate'
                  title={title}
                >
                  {title}
                </span>
                <Button
                  variant='ghost'
                  size='icon'
                  onClick={(event) => {
                    event.stopPropagation();
                    onDeleteConversation(conversation.id);
                  }}
                  aria-label='Excluir chat'
                  className='ml-2 h-7 w-7 text-red-500 bg-red-500/20 hover:bg-red-500/30 hover:text-red-400 flex-shrink-0'
                >
                  <Trash2 className='h-4 w-4' />
                </Button>
              </div>
            );
          })}
        </div>
      </ScrollArea>

      <div className='p-3 border-t border-sidebar-border'>
        <div className='flex items-center gap-3'>
          <Avatar className='h-8 w-8'>
            <AvatarFallback className='bg-sidebar-accent text-sidebar-foreground text-xs'>
              {user ? getUserInitials(user.email) : 'U'}
            </AvatarFallback>
          </Avatar>
          <div className='flex-1 min-w-0'>
            <p className='text-sm text-sidebar-foreground truncate'>
              {user?.email}
            </p>
          </div>
          <Button
            variant='ghost'
            size='icon'
            onClick={logout}
            className='text-sidebar-foreground hover:bg-sidebar-accent flex-shrink-0'
          >
            <LogOut className='h-4 w-4' />
          </Button>
        </div>
      </div>
    </aside>
  );
}
