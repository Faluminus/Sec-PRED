import { Toaster, toast } from 'sonner'


export default function Component() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-sky-50 p-4">
      <Toaster
        theme="light"
        position="top-right"
        toastOptions={{
          style: {
            background: 'white',
            color: '#0369a1',
            border: '1px solid #e0f2fe',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          },
        }}
      />
      <div className="space-y-4">
        <button 
          onClick={() => toast.info('Here s some information')}
          className="bg-sky-600 hover:bg-sky-700 text-white"
        >
          Show Info Toast
        </button>
        <button 
          onClick={() => 
            toast('Custom Toast', {
              icon: '🚀',
              description: 'This is a custom toast notification',
            })
          }
          className="bg-sky-600 hover:bg-sky-700 text-white"
        >
          Show Custom Toast
        </button>
      </div>
    </div>
  )
}