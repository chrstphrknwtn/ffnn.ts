export interface Config {
  width: number
  height: number
  input: number
  hidden: number
  maxR: number
  maxG: number
  maxB: number
}

interface Model {
  w_in: Matrix
  w_out: Matrix
  [key: string]: any
}

/*
 * Activation functions
 */
function tanh(matrix: Matrix): Matrix {
  const out = new Matrix(matrix.n, matrix.d)
  const n = matrix.w.length
  for (let i = 0; i < n; i++) {
    out.w[i] = Math.tanh(matrix.w[i])
  }
  return out
}

function sigmoid(matrix: Matrix): Matrix {
  const out = new Matrix(matrix.n, matrix.d)
  const n = matrix.w.length
  for (let i = 0; i < n; i++) {
    out.w[i] = sig(matrix.w[i])
  }
  return out
}

function sig(x: number): number {
  return 1.0 / (1 + Math.exp(-x))
}

/*
 * Matrices
 */

function matrixMul(m1: Matrix, m2: Matrix): Matrix {
  if (m1.d !== m2.n) {
    throw Error('Matrix dimensions misaligned')
  }

  const n = m1.n
  const d = m2.d
  const out = new Matrix(n, d)
  const m1d = m1.d
  const m2d = m2.d

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      let dot = 0.0
      for (let k = 0; k < m1d; k++) {
        dot += m1.w[m1d * i + k] * m2.w[m2d * k + j]
      }
      out.w[d * i + j] = dot
    }
  }

  return out
}

function zeros(num: number): Float64Array {
  return isNaN(num) ? new Float64Array() : new Float64Array(num)
}

function generateGaussianRandom(): number {
  let u, v, r, c
  do {
    u = 2 * Math.random() - 1
    v = 2 * Math.random() - 1
    r = u * u + v * v
  } while (r === 0 || r > 1)
  c = Math.sqrt((-2 * Math.log(r)) / r)
  return u * c
}

const generateRandomNormal = (
  mean: number,
  standardDeviation: number
): number => {
  return mean + generateGaussianRandom() * standardDeviation
}

function fillRandn(
  matrix: Matrix,
  mean: number,
  standardDeviation: number
): void {
  for (let i = 0, n = matrix.w.length; i < n; i++) {
    matrix.w[i] = generateRandomNormal(mean, standardDeviation)
  }
}

function RandMat(
  rows: number,
  columns: number,
  mean: number,
  standardDeviation: number
): Matrix {
  const matrix = new Matrix(rows, columns)
  fillRandn(matrix, mean || 0, standardDeviation || 0.08)
  return matrix
}

/*
Matrix Class
*/

class Matrix {
  n: number
  d: number
  w: Float64Array

  constructor(n: number, d: number) {
    this.n = n
    this.d = d
    this.w = zeros(n * d)
  }

  get(row: number, col: number) {
    const ix = this.d * row + col
    if (ix >= 0 && ix > this.w.length) {
      throw Error('index out of bounds')
    }
    return this.w[ix]
  }

  set(row: number, col: number, value: number) {
    const ix = this.d * row + col
    if (ix >= 0 && ix > this.w.length) {
      throw Error('index out of bounds')
    }
    this.w[ix] = value
  }
}

const createModel = (config: Config): Model => {
  const weightInitRange = 0.8

  const model: Model = {
    w_in: RandMat(config.input, 3, 0, weightInitRange),
    w_out: RandMat(3, config.input, 0, weightInitRange)
  }

  // Initialize weights for the hidden layers
  for (let layerIndex = 0; layerIndex < config.hidden; layerIndex++) {
    model[`w_${layerIndex}`] = RandMat(
      config.input,
      config.input,
      0,
      weightInitRange
    )
  }

  return model
}

const forwardNetwork = (
  config: Config,
  model: Model,
  x_: number,
  y_: number
): Matrix => {
  const x = new Matrix(3, 1)

  x.set(0, 0, x_)
  x.set(1, 0, y_)
  x.set(2, 0, 1.0) // bias

  let out = tanh(matrixMul(model.w_in, x))
  for (let i = 0; i < config.hidden; i++) {
    out = tanh(matrixMul(model[`w_${i}`], out))
  }
  out = sigmoid(matrixMul(model.w_out, out))

  return out
}

const getColorAt = (
  config: Config,
  model: Model,
  x: number,
  y: number
): number[] => {
  const noiseRange = 0.03
  const noise = () => (Math.random() * 2 - 1) * noiseRange
  const out = forwardNetwork(config, model, x, y)

  const r = (out.w[0] + noise()) * config.maxR
  const g = (out.w[1] + noise()) * config.maxG
  const b = (out.w[2] + noise()) * config.maxB
  const a = 255
  return [
    Math.min(255, Math.max(0, r)),
    Math.min(255, Math.max(0, g)),
    Math.min(255, Math.max(0, b)),
    a
  ]
}

const getImageData = (config: Config): Uint8ClampedArray => {
  const model = createModel(config)
  const pixels = new Uint8ClampedArray(config.width * config.height * 4)

  for (let y = 0; y < config.height; y++) {
    for (let x = 0; x < config.width; x++) {
      const [r, g, b, a] = getColorAt(
        config,
        model,
        x / config.width,
        y / config.height
      )
      const idx = (y * config.width + x) * 4
      pixels[idx] = r
      pixels[idx + 1] = g
      pixels[idx + 2] = b
      pixels[idx + 3] = a
    }
  }

  return pixels
}

export default getImageData
