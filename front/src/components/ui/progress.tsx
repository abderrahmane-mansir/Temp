import { cn } from "@/lib/utils"

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value: number;
  max?: number;
  variant?: "default" | "success" | "destructive";
}

export function Progress({ value, max = 100, variant = "default", className, ...props }: ProgressProps) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  
  return (
    <div
      className={cn("relative h-2 w-full overflow-hidden rounded-full bg-secondary", className)}
      {...props}
    >
      <div
        className={cn(
          "h-full transition-all duration-500 ease-out rounded-full",
          {
            "bg-primary": variant === "default",
            "bg-success": variant === "success",
            "bg-destructive": variant === "destructive",
          }
        )}
        style={{ width: `${percentage}%` }}
      />
    </div>
  )
}
