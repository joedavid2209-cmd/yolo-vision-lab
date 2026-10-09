import { useState, useRef, useCallback, useEffect } from 'react'
import { YoloDetector, drawBoxes, type BoundingBox } from './yolo'

// ── Icons (inline SVG components) ──────────────────────────────────────────

function IconCamera({ className = '' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
      <circle cx="12" cy="13" r="3" />
    </svg>
  )
}

function IconCpu({ className = '' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <rect x="9" y="9" width="6" height="6" />
      <line x1="9" y1="1" x2="9" y2="4" />
      <line x1="15" y1="1" x2="15" y2="4" />
      <line x1="9" y1="20" x2="9" y2="23" />
      <line x1="15" y1="20" x2="15" y2="23" />
      <line x1="20" y1="9" x2="23" y2="9" />
      <line x1="20" y1="14" x2="23" y2="14" />
      <line x1="1" y1="9" x2="4" y2="9" />
      <line x1="1" y1="14" x2="4" y2="14" />
    </svg>
  )
}

function IconBox({ className = '' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="6" height="6" rx="1" />
      <rect x="15" y="3" width="6" height="6" rx="1" />
      <rect x="3" y="15" width="6" height="6" rx="1" />
      <rect x="15" y="15" width="6" height="6" rx="1" />
      <line x1="9" y1="6" x2="15" y2="6" />
      <line x1="6" y1="9" x2="6" y2="15" />
      <line x1="9" y1="18" x2="15" y2="18" />
      <line x1="18" y1="9" x2="18" y2="15" />
    </svg>
  )
}

function IconVideoOff({ className = '' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M16 16v1a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h2m5.66 0H14a2 2 0 0 1 2 2v3.34l1 1L23 7v10" />
      <line x1="1" y1="1" x2="23" y2="23" />
    </svg>
  )
}

// ── Types ──────────────────────────────────────────────────────────────────

interface Detection {
  id: number
  label: string
  confidence: number
}

// ── Stat Card ──────────────────────────────────────────────────────────────

function StatCard({ label, value, valueClass = 'text-white' }: {
  label: string
  value: string
  valueClass?: string
}) {
  return (
    <div className="flex-1 bg-[#111827] rounded-xl p-4 border border-[#1F2937] flex flex-col gap-1 min-w-0">
      <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">{label}</span>
      <span className={`text-2xl font-bold font-mono stat-number ${valueClass}`}>{value}</span>
    </div>
  )
}

// ── Pipeline Step Card ─────────────────────────────────────────────────────

function StepCard({ step, icon, title, desc }: {
  step: string
  icon: React.ReactNode
  title: string
  desc: string
}) {
  return (
    <div className="flex-1 bg-[#111827] rounded-xl p-5 border border-[#1F2937] flex flex-col gap-3 min-w-[200px] hover:border-[#0F766E]/40 transition-colors duration-300">
      <div className="flex items-center gap-3">
        <span className="text-[10px] font-mono font-semibold text-[#0F766E] bg-[#0F766E]/10 px-2 py-0.5 rounded border border-[#0F766E]/20">
          {step}
        </span>
        <div className="text-[#0F766E]">{icon}</div>
      </div>
      <div>
        <p className="font-semibold text-white text-sm mb-1">{title}</p>
        <p className="text-gray-400 text-xs leading-relaxed">{desc}</p>
      </div>
    </div>
  )
}

// ── Main App ───────────────────────────────────────────────────────────────

export default function App() {
  const [isRunning, setIsRunning] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [fps, setFps] = useState<number | null>(null)
  const [detections, setDetections] = useState<Detection[]>([])
  const [confidence, setConfidence] = useState(0.5)

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const detectorRef = useRef<YoloDetector | null>(null)
  const runningRef = useRef(false)
  const confidenceRef = useRef(confidence)

  // Keep confidence ref in sync so the detection loop always uses latest value
  useEffect(() => {
    confidenceRef.current = confidence
  }, [confidence])

  // ── Detection loop ──
  const runDetectionLoop = useCallback(async () => {
    const detector = detectorRef.current
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!detector || !video || !canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let lastTime = performance.now()
    let frameCount = 0

    const loop = async () => {
      if (!runningRef.current) return

      const startTime = performance.now()

      // Run inference
      const boxes: BoundingBox[] = await detector.detect(video, confidenceRef.current)

      // Sync canvas size to video display size
      const rect = video.getBoundingClientRect()
      if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width
        canvas.height = rect.height
      }

      // Scale boxes from video resolution to canvas display size
      const scaleX = rect.width / video.videoWidth
      const scaleY = rect.height / video.videoHeight
      const scaledBoxes = boxes.map((b) => ({
        ...b,
        x: b.x * scaleX,
        y: b.y * scaleY,
        w: b.w * scaleX,
        h: b.h * scaleY,
      }))

      // Draw bounding boxes
      drawBoxes(ctx, scaledBoxes)

      // Update detections list
      const newDetections: Detection[] = boxes.map((b, i) => ({
        id: i,
        label: b.label,
        confidence: b.score,
      }))
      setDetections(newDetections)

      // Calculate FPS
      frameCount++
      const elapsed = performance.now() - lastTime
      if (elapsed >= 1000) {
        setFps(Math.round((frameCount * 1000) / elapsed))
        frameCount = 0
        lastTime = performance.now()
      }

      // Schedule next frame — use setTimeout to avoid blocking the main thread
      const inferenceTime = performance.now() - startTime
      const delay = Math.max(1, 33 - inferenceTime) // target ~30fps max
      setTimeout(loop, delay)
    }

    loop()
  }, [])

  const startDetection = useCallback(async () => {
    try {
      setIsLoading(true)

      // Request webcam
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      // Load YOLO model (first time only — takes a few seconds)
      if (!detectorRef.current) {
        detectorRef.current = new YoloDetector()
        await detectorRef.current.load('/yolov8n.onnx')
      }

      setIsLoading(false)
      setIsRunning(true)
      runningRef.current = true

      // Start the detection loop
      runDetectionLoop()
    } catch (err) {
      setIsLoading(false)
      console.error('Failed to start detection:', err)
    }
  }, [runDetectionLoop])

  const stopDetection = useCallback(() => {
    runningRef.current = false

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
    setIsRunning(false)
    setFps(null)
    setDetections([])
  }, [])

  const toggleDetection = () => {
    if (isRunning) stopDetection()
    else startDetection()
  }

  const objectCount = detections.length

  return (
    <div className="min-h-screen bg-[#0A0F1A] text-white font-sans">
      {/* ── Subtle grid overlay ── */}
      <div
        className="fixed inset-0 pointer-events-none opacity-[0.03]"
        style={{
          backgroundImage: 'linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px)',
          backgroundSize: '48px 48px',
        }}
      />

      <div className="relative mx-auto max-w-[1024px] px-4 sm:px-6 py-8 flex flex-col gap-6">

        {/* ── Header ── */}
        <header className="flex items-start gap-4">
          <div className="shrink-0 w-9 h-9 rounded-lg bg-[#0F766E]/20 border border-[#0F766E]/30 flex items-center justify-center">
            <span className="text-[#14B8A6] font-bold text-sm tracking-tight">JD</span>
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-3 flex-wrap">
              <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
                Vision Lab
              </h1>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-[#0F766E]/15 border border-[#0F766E]/30 text-[#14B8A6]">
                <span className="w-1.5 h-1.5 rounded-full bg-[#14B8A6] live-dot" />
                Live Demo
              </span>
            </div>
            <p className="mt-1 text-sm text-gray-400 leading-snug">
              Real-time object detection running entirely in your browser.
            </p>
          </div>
        </header>

        {/* ── Webcam Viewport ── */}
        <div className="relative w-full" style={{ paddingBottom: '56.25%' }}>
          <div className="absolute inset-0 rounded-2xl overflow-hidden border border-[#0F766E]/40 bg-[#111827] shadow-[0_0_40px_-8px_rgba(15,118,110,0.25)]">
            <video
              ref={videoRef}
              className="absolute inset-0 w-full h-full object-cover"
              muted
              playsInline
            />

            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />

            {/* Placeholder — shown when not running */}
            {!isRunning && !isLoading && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
                <div className="w-16 h-16 rounded-2xl bg-[#0F766E]/10 border border-[#0F766E]/20 flex items-center justify-center">
                  <IconVideoOff className="w-8 h-8 text-gray-600" />
                </div>
                <div className="text-center">
                  <p className="text-gray-400 font-medium">Click <span className="text-[#14B8A6]">Start</span> to begin detection</p>
                  <p className="text-gray-600 text-sm mt-1">Camera access required</p>
                </div>
              </div>
            )}

            {/* Loading state */}
            {isLoading && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-[#111827]/80">
                <div className="w-10 h-10 border-2 border-[#0F766E] border-t-transparent rounded-full animate-spin" />
                <p className="text-gray-400 text-sm font-medium">Loading YOLOv8n model...</p>
              </div>
            )}

            {/* Scanning line — shown when running */}
            {isRunning && (
              <div
                className="scan-line absolute left-0 right-0 h-px pointer-events-none"
                style={{
                  background: 'linear-gradient(90deg, transparent 0%, #14B8A6 20%, #0F766E 50%, #14B8A6 80%, transparent 100%)',
                  boxShadow: '0 0 8px 1px rgba(20,184,166,0.4)',
                }}
              />
            )}

            {/* Active indicator */}
            {isRunning && (
              <div className="absolute top-3 left-3 flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur-sm border border-white/10 text-xs font-medium text-white">
                <span className="w-1.5 h-1.5 rounded-full bg-red-500 live-dot" />
                REC
              </div>
            )}

            {/* FPS badge */}
            {isRunning && fps !== null && (
              <div className="absolute top-3 right-3 px-2 py-1 rounded bg-black/60 backdrop-blur-sm border border-white/10 text-xs font-mono text-[#14B8A6]">
                {fps} FPS
              </div>
            )}

            {/* Corner brackets */}
            <div className="absolute top-3 left-3 w-5 h-5 border-t-2 border-l-2 border-[#0F766E]/60 rounded-tl pointer-events-none" />
            <div className="absolute top-3 right-3 w-5 h-5 border-t-2 border-r-2 border-[#0F766E]/60 rounded-tr pointer-events-none" />
            <div className="absolute bottom-3 left-3 w-5 h-5 border-b-2 border-l-2 border-[#0F766E]/60 rounded-bl pointer-events-none" />
            <div className="absolute bottom-3 right-3 w-5 h-5 border-b-2 border-r-2 border-[#0F766E]/60 rounded-br pointer-events-none" />
          </div>
        </div>

        {/* ── Controls Bar ── */}
        <div className="flex flex-wrap items-center gap-4 bg-[#111827] rounded-xl px-5 py-4 border border-[#1F2937]">
          <button
            onClick={toggleDetection}
            disabled={isLoading}
            className={`
              relative flex items-center gap-2 px-6 py-2.5 rounded-lg font-semibold text-sm
              transition-all duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-[#111827]
              disabled:opacity-50 disabled:cursor-not-allowed
              ${isRunning
                ? 'bg-red-600 hover:bg-red-500 text-white focus:ring-red-500 shadow-[0_0_20px_-4px_rgba(220,38,38,0.5)]'
                : 'bg-[#0F766E] hover:bg-[#14B8A6] text-white focus:ring-[#0F766E] shadow-[0_0_20px_-4px_rgba(15,118,110,0.5)]'
              }
            `}
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Loading...
              </>
            ) : isRunning ? (
              <>
                <span className="w-3 h-3 rounded-sm bg-white/90 shrink-0" />
                Stop Detection
              </>
            ) : (
              <>
                <IconCamera className="w-4 h-4 shrink-0" />
                Start Detection
              </>
            )}
          </button>

          <div className="hidden sm:block w-px h-8 bg-[#1F2937]" />

          <div className="flex items-center gap-3 flex-1 min-w-[200px]">
            <span className="text-xs text-gray-500 shrink-0 font-medium">Confidence</span>
            <input
              type="range"
              min={0.25}
              max={0.75}
              step={0.01}
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
              className="flex-1 cursor-pointer"
            />
            <span className="text-xs font-mono font-semibold text-[#14B8A6] w-10 text-right shrink-0">
              {confidence.toFixed(2)}
            </span>
          </div>
        </div>

        {/* ── Stats Panel ── */}
        <div className="flex gap-3 flex-wrap sm:flex-nowrap">
          <StatCard
            label="FPS"
            value={fps !== null ? String(fps) : '—'}
            valueClass={fps !== null ? 'text-[#14B8A6]' : 'text-gray-600'}
          />
          <StatCard
            label="Objects detected"
            value={String(objectCount)}
            valueClass={objectCount > 0 ? 'text-[#F59E0B]' : 'text-white'}
          />
          <StatCard
            label="Model"
            value="YOLOv8n"
            valueClass="text-[#0F766E]"
          />
        </div>

        {/* ── Detections List ── */}
        <div className="bg-[#111827] rounded-xl border border-[#1F2937] overflow-hidden">
          <div className="flex items-center justify-between px-5 py-3.5 border-b border-[#1F2937]">
            <h2 className="text-sm font-semibold text-white tracking-tight">Detected Objects</h2>
            {detections.length > 0 && (
              <span className="text-xs font-mono text-gray-500">{detections.length} found</span>
            )}
          </div>
          <div className="px-5 py-4 max-h-40 overflow-y-auto scrollbar-hide">
            {detections.length === 0 ? (
              <p className="text-sm text-gray-600 text-center py-2">No detections yet</p>
            ) : (
              <div className="flex flex-wrap gap-2">
                {detections.map((d) => (
                  <span
                    key={d.id}
                    className="detection-pill inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-[#0F766E]/20 border border-[#0F766E]/30 text-[#14B8A6]"
                  >
                    {d.label}
                    <span className="text-[#0F766E] font-mono">{(d.confidence * 100).toFixed(0)}%</span>
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ── How It Works ── */}
        <div>
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-widest mb-3">How it works</h2>
          <div className="flex flex-wrap gap-3 sm:flex-nowrap">
            <StepCard
              step="01"
              icon={<IconCamera className="w-5 h-5" />}
              title="Capture"
              desc="Webcam feeds live frames to the model at up to 30 FPS via the MediaDevices API."
            />
            <StepCard
              step="02"
              icon={<IconCpu className="w-5 h-5" />}
              title="Detect"
              desc="YOLOv8n identifies 80 object classes via ONNX Runtime Web — no server round-trips."
            />
            <StepCard
              step="03"
              icon={<IconBox className="w-5 h-5" />}
              title="Render"
              desc="Bounding boxes and labels are drawn in real time on a canvas overlay."
            />
          </div>
        </div>

        {/* ── Footer ── */}
        <footer className="flex flex-col gap-4 pt-2 border-t border-[#1F2937]">
          <div className="flex flex-wrap gap-2">
            {['YOLOv8n', 'ONNX Runtime Web', 'React', 'TypeScript', 'Tailwind CSS', 'WebAssembly'].map((tag) => (
              <span
                key={tag}
                className="text-xs px-2.5 py-1 rounded-full bg-[#1F2937] border border-[#374151] text-gray-400 font-medium"
              >
                {tag}
              </span>
            ))}
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 text-xs text-gray-600">
            <span>
              Built by Joe David ·{' '}
              <a
                href="https://joedavid.pro"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#0F766E] hover:text-[#14B8A6] transition-colors underline underline-offset-2"
              >
                joedavid.pro
              </a>
            </span>
            <span className="text-gray-700">
              All processing happens locally — no data leaves your device.
            </span>
          </div>
        </footer>
      </div>
    </div>
  )
}
