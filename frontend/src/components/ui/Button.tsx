import { ButtonHTMLAttributes, ReactNode } from 'react'
import clsx from 'clsx'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger'
  size?: 'sm' | 'md' | 'lg'
  children: ReactNode
}

export default function Button({
  variant = 'primary',
  size = 'md',
  className,
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      className={clsx(
        'rounded-lg font-medium transition-colors',
        {
          // Primary: 蓝色边框白底
          'border border-blue-500 bg-white text-blue-600 hover:border-blue-600 hover:bg-blue-50 disabled:opacity-50 disabled:cursor-not-allowed': variant === 'primary',
          // Secondary: 灰色边框白底
          'border border-slate-300 bg-white text-slate-700 hover:border-slate-400 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed': variant === 'secondary',
          // Danger: 红色边框白底
          'border border-red-500 bg-white text-red-600 hover:border-red-600 hover:bg-red-50 disabled:opacity-50 disabled:cursor-not-allowed': variant === 'danger',
          'px-3 py-1.5 text-sm': size === 'sm',
          'px-4 py-2 text-sm': size === 'md',
          'px-6 py-2.5 text-base': size === 'lg',
        },
        className
      )}
      {...props}
    >
      {children}
    </button>
  )
}
