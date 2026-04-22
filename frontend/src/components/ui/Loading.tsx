export default function Loading({ message = '加载中...' }: { message?: string }) {
  return (
    <div className="fixed inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-white rounded-lg border border-gray-200 p-6 text-center">
        <div className="flex gap-2 mb-4 justify-center">
          <span className="w-3 h-3 bg-gray-800 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></span>
          <span className="w-3 h-3 bg-gray-800 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
          <span className="w-3 h-3 bg-gray-800 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></span>
        </div>
        <p className="text-gray-700 font-medium">{message}</p>
      </div>
    </div>
  )
}
