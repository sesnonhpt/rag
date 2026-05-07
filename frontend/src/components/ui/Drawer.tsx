import { ReactNode, useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import clsx from 'clsx'

interface DrawerProps {
  open: boolean
  onClose: () => void
  title?: string
  children: ReactNode
  width?: 'sm' | 'md' | 'lg' | 'xl'
  position?: 'left' | 'right'
}

export default function Drawer({
  open,
  onClose,
  title,
  children,
  width = 'lg',
  position = 'right',
}: DrawerProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [isAnimating, setIsAnimating] = useState(false)

  useEffect(() => {
    if (open) {
      setIsVisible(true)
      // Trigger animation after render
      requestAnimationFrame(() => {
        setIsAnimating(true)
      })
    } else {
      setIsAnimating(false)
      // Wait for animation to complete before hiding
      const timer = setTimeout(() => {
        setIsVisible(false)
      }, 300) // Match transition duration
      return () => clearTimeout(timer)
    }
  }, [open])

  // Handle ESC key
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && open) {
        onClose()
      }
    }
    window.addEventListener('keydown', handleEsc)
    return () => window.removeEventListener('keydown', handleEsc)
  }, [open, onClose])

  if (!isVisible) return null

  const widthClasses = {
    sm: 'w-80',
    md: 'w-96',
    lg: 'w-[480px]',
    xl: 'w-[600px]',
  }

  const positionClasses = {
    left: 'left-0',
    right: 'right-0',
  }

  const slideClasses = {
    left: isAnimating ? 'translate-x-0' : '-translate-x-full',
    right: isAnimating ? 'translate-x-0' : 'translate-x-full',
  }

  return createPortal(
    <>
      {/* Drawer - No backdrop */}
      <div
        className={clsx(
          'fixed top-0 bottom-0 bg-white border-l border-gray-200 shadow-2xl transition-transform duration-300 ease-out z-50 flex flex-col',
          widthClasses[width],
          positionClasses[position],
          slideClasses[position]
        )}
      >
        {/* Header */}
        {title && (
          <div className="flex items-center justify-between border-b border-gray-200 px-6 py-4 bg-white">
            <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
            <button
              onClick={onClose}
              className="rounded-lg p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600 transition-colors"
            >
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-y-auto bg-gray-50">
          {children}
        </div>
      </div>
    </>,
    document.body
  )
}
