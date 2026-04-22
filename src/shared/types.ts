// 共享类型定义

export interface DataFiles {
  eegFiles: string[]
  labelFiles: string[]
  total: number
}

export interface ProcessResult {
  success: boolean
  output: string
  code?: number
}

export interface TestResult extends ProcessResult {
  resultsPath?: string
}

export interface EEGWaveData {
  success: boolean
  data?: number[]
  startTime?: number
  duration?: number
  sampleRate?: number
  error?: string
}

export interface ResultsData {
  success: boolean
  results?: Record<string, string>
  resultsPath?: string
  error?: string
}
