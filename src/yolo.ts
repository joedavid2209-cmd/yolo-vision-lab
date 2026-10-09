import * as ort from 'onnxruntime-web'

// ── COCO class names (80 classes) ─────────────────────────────────────────
const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
  'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
  'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush',
]

// ── Types ─────────────────────────────────────────────────────────────────
export interface BoundingBox {
  x: number
  y: number
  w: number
  h: number
  label: string
  score: number
  classId: number
}

// ── Model dimensions ──────────────────────────────────────────────────────
const MODEL_WIDTH = 640
const MODEL_HEIGHT = 640

// ── Preprocessing: video frame → ONNX tensor ─────────────────────────────
function preprocess(
  ctx: CanvasRenderingContext2D,
  video: HTMLVideoElement,
): ort.Tensor {
  // Draw video frame onto a 640x640 canvas
  ctx.drawImage(video, 0, 0, MODEL_WIDTH, MODEL_HEIGHT)
  const imageData = ctx.getImageData(0, 0, MODEL_WIDTH, MODEL_HEIGHT)
  const { data } = imageData

  // Convert RGBA → RGB float32, normalize to [0,1], arrange as NCHW
  const totalPixels = MODEL_WIDTH * MODEL_HEIGHT
  const float32Data = new Float32Array(3 * totalPixels)

  for (let i = 0; i < totalPixels; i++) {
    const rgbaIdx = i * 4
    float32Data[i] = data[rgbaIdx] / 255                    // R channel
    float32Data[totalPixels + i] = data[rgbaIdx + 1] / 255  // G channel
    float32Data[2 * totalPixels + i] = data[rgbaIdx + 2] / 255 // B channel
  }

  return new ort.Tensor('float32', float32Data, [1, 3, MODEL_HEIGHT, MODEL_WIDTH])
}

// ── Non-Maximum Suppression ───────────────────────────────────────────────
function nms(boxes: BoundingBox[], iouThreshold = 0.45): BoundingBox[] {
  // Sort by score descending
  const sorted = [...boxes].sort((a, b) => b.score - a.score)
  const kept: BoundingBox[] = []

  const suppressed = new Set<number>()

  for (let i = 0; i < sorted.length; i++) {
    if (suppressed.has(i)) continue
    kept.push(sorted[i])

    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed.has(j)) continue
      if (iou(sorted[i], sorted[j]) > iouThreshold) {
        suppressed.add(j)
      }
    }
  }

  return kept
}

function iou(a: BoundingBox, b: BoundingBox): number {
  const x1 = Math.max(a.x, b.x)
  const y1 = Math.max(a.y, b.y)
  const x2 = Math.min(a.x + a.w, b.x + b.w)
  const y2 = Math.min(a.y + a.h, b.y + b.h)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const union = a.w * a.h + b.w * b.h - inter
  return union > 0 ? inter / union : 0
}

// ── Postprocess: ONNX output → bounding boxes ────────────────────────────
function postprocess(
  output: ort.Tensor,
  confThreshold: number,
  videoWidth: number,
  videoHeight: number,
): BoundingBox[] {
  // YOLOv8 output shape: [1, 84, 8400]
  // 84 = 4 (cx, cy, w, h) + 80 (class scores)
  // We need to transpose to [8400, 84]
  const data = output.data as Float32Array
  const numDetections = 8400
  const numClasses = 80

  const scaleX = videoWidth / MODEL_WIDTH
  const scaleY = videoHeight / MODEL_HEIGHT

  const boxes: BoundingBox[] = []

  for (let i = 0; i < numDetections; i++) {
    // Extract box coordinates (stored column-major in [1, 84, 8400])
    const cx = data[0 * numDetections + i]
    const cy = data[1 * numDetections + i]
    const w = data[2 * numDetections + i]
    const h = data[3 * numDetections + i]

    // Find best class
    let maxScore = 0
    let maxClassId = 0
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numDetections + i]
      if (score > maxScore) {
        maxScore = score
        maxClassId = c
      }
    }

    if (maxScore < confThreshold) continue

    // Convert from center-format to top-left-format, scale to video size
    boxes.push({
      x: (cx - w / 2) * scaleX,
      y: (cy - h / 2) * scaleY,
      w: w * scaleX,
      h: h * scaleY,
      label: COCO_CLASSES[maxClassId] ?? `class_${maxClassId}`,
      score: maxScore,
      classId: maxClassId,
    })
  }

  return nms(boxes)
}

// ── Draw bounding boxes on canvas ─────────────────────────────────────────
// Assign a stable color per class
const CLASS_COLORS: Record<number, string> = {}
function getClassColor(classId: number): string {
  if (!CLASS_COLORS[classId]) {
    const hue = (classId * 137.508) % 360 // golden angle for nice distribution
    CLASS_COLORS[classId] = `hsl(${hue}, 70%, 55%)`
  }
  return CLASS_COLORS[classId]
}

export function drawBoxes(
  ctx: CanvasRenderingContext2D,
  boxes: BoundingBox[],
): void {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

  for (const box of boxes) {
    const color = getClassColor(box.classId)

    // Box outline
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.strokeRect(box.x, box.y, box.w, box.h)

    // Label background
    const labelText = `${box.label} ${(box.score * 100).toFixed(0)}%`
    ctx.font = 'bold 13px Inter, system-ui, sans-serif'
    const textWidth = ctx.measureText(labelText).width
    const labelHeight = 22
    const labelY = box.y > labelHeight ? box.y - labelHeight : box.y

    ctx.fillStyle = color
    ctx.fillRect(box.x, labelY, textWidth + 10, labelHeight)

    // Label text
    ctx.fillStyle = '#fff'
    ctx.fillText(labelText, box.x + 5, labelY + 15)
  }
}

// ── YOLO session manager ──────────────────────────────────────────────────
export class YoloDetector {
  private session: ort.InferenceSession | null = null
  private preprocessCanvas: HTMLCanvasElement
  private preprocessCtx: CanvasRenderingContext2D

  constructor() {
    // Off-screen canvas for preprocessing (always 640x640)
    this.preprocessCanvas = document.createElement('canvas')
    this.preprocessCanvas.width = MODEL_WIDTH
    this.preprocessCanvas.height = MODEL_HEIGHT
    this.preprocessCtx = this.preprocessCanvas.getContext('2d', {
      willReadFrequently: true,
    })!
  }

  async load(modelUrl = '/yolov8n.onnx'): Promise<void> {
    // Use WASM backend (works in all browsers, no GPU needed)
    ort.env.wasm.numThreads = 1
    ort.env.wasm.simd = true

    this.session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
    })
  }

  async detect(
    video: HTMLVideoElement,
    confThreshold: number,
  ): Promise<BoundingBox[]> {
    if (!this.session) return []

    // Preprocess
    const inputTensor = preprocess(this.preprocessCtx, video)

    // Run inference
    const feeds: Record<string, ort.Tensor> = { images: inputTensor }
    const results = await this.session.run(feeds)

    // Get the output tensor (YOLOv8 uses "output0")
    const outputTensor = results['output0']
    if (!outputTensor) return []

    // Postprocess
    return postprocess(
      outputTensor,
      confThreshold,
      video.videoWidth,
      video.videoHeight,
    )
  }

  dispose(): void {
    this.session?.release()
    this.session = null
  }
}
