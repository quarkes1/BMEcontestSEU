import { Chart, registerables } from 'chart.js'

// 注册 Chart.js 组件
Chart.register(...registerables)

// TypeScript 接口定义
interface ElectronAPI {
  selectTrainData: () => Promise<string | null>
  selectTestData: () => Promise<string | null>
  getDataFiles: (dataPath: string) => Promise<{ eegFiles: string[], labelFiles: string[], total: number } | null>
  runPreprocessing: (trainDataPath: string) => Promise<{ success: boolean, output: string }>
  runTraining: () => Promise<{ success: boolean, output: string }>
  runTesting: (testDataPath: string) => Promise<{ success: boolean, output: string, resultsPath: string }>
  readEEGWave: (filePath: string, startSec: number, durationSec: number) => Promise<{ success: boolean, data: number[], startTime: number, duration: number, sampleRate: number }>
  readResults: () => Promise<{ success: boolean, results: any, resultsPath: string }>
  getResultImages: () => Promise<{ success: boolean, images: Array<{ name: string, path: string }> }>
  stopProcess: () => Promise<boolean>
  checkModel: () => Promise<boolean>
  checkProcessedData: () => Promise<boolean>
  onPythonLog: (callback: (log: string) => void) => void
  removePythonLogListener: () => void
}

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
}

// 应用状态
interface AppState {
  trainDataPath: string
  testDataPath: string
  eegFiles: string[]
  labelFiles: string[]
  isProcessing: boolean
  modelReady: boolean
  processedDataReady: boolean
  waveformChart: Chart | null
}

const state: AppState = {
  trainDataPath: '',
  testDataPath: '',
  eegFiles: [],
  labelFiles: [],
  isProcessing: false,
  modelReady: false,
  processedDataReady: false,
  waveformChart: null
}

// DOM 元素
const elements = {
  trainDataPath: document.getElementById('trainDataPath') as HTMLInputElement,
  testDataPath: document.getElementById('testDataPath') as HTMLInputElement,
  selectTrainBtn: document.getElementById('selectTrainBtn') as HTMLButtonElement,
  selectTestBtn: document.getElementById('selectTestBtn') as HTMLButtonElement,
  runPreprocessBtn: document.getElementById('runPreprocessBtn') as HTMLButtonElement,
  runTrainingBtn: document.getElementById('runTrainingBtn') as HTMLButtonElement,
  runTestingBtn: document.getElementById('runTestingBtn') as HTMLButtonElement,
  stopBtn: document.getElementById('stopBtn') as HTMLButtonElement,
  trainDataInfo: document.getElementById('trainDataInfo') as HTMLDivElement,
  preprocessStatus: document.getElementById('preprocessStatus') as HTMLDivElement,
  trainingStatus: document.getElementById('trainingStatus') as HTMLDivElement,
  testingStatus: document.getElementById('testingStatus') as HTMLDivElement,
  modelInfo: document.getElementById('modelInfo') as HTMLDivElement,
  eegFileSelect: document.getElementById('eegFileSelect') as HTMLSelectElement,
  loadWaveBtn: document.getElementById('loadWaveBtn') as HTMLButtonElement,
  randomWaveBtn: document.getElementById('randomWaveBtn') as HTMLButtonElement,
  waveformInfo: document.getElementById('waveformInfo') as HTMLDivElement,
  logContainer: document.getElementById('logContainer') as HTMLDivElement,
  clearLogBtn: document.getElementById('clearLogBtn') as HTMLButtonElement,
  exportLogBtn: document.getElementById('exportLogBtn') as HTMLButtonElement,
  resultsContainer: document.getElementById('resultsContainer') as HTMLDivElement,
  statusText: document.getElementById('statusText') as HTMLSpanElement,
  tabs: document.querySelectorAll('.tab'),
  tabContents: document.querySelectorAll('.tab-content')
}

// 初始化
async function init() {
  // 检查模型和预处理数据状态
  state.modelReady = await window.electronAPI.checkModel()
  state.processedDataReady = await window.electronAPI.checkProcessedData()
  
  updateButtonStates()
  
  // 监听 Python 日志
  window.electronAPI.onPythonLog((log: string) => {
    appendLog(log)
  })
  
  // 绑定事件
  bindEvents()
  
  updateStatus('就绪')
}

// 绑定事件
function bindEvents() {
  // 选择训练数据
  elements.selectTrainBtn.addEventListener('click', async () => {
    const path = await window.electronAPI.selectTrainData()
    if (path) {
      state.trainDataPath = path
      elements.trainDataPath.value = path
      
      // 获取文件列表
      const files = await window.electronAPI.getDataFiles(path)
      if (files) {
        state.eegFiles = files.eegFiles
        state.labelFiles = files.labelFiles
        
        elements.trainDataInfo.innerHTML = `
          <div>脑电文件: ${files.eegFiles.length} 个</div>
          <div>标签文件: ${files.labelFiles.length} 个</div>
        `
        
        // 更新下拉列表
        updateEEGFileSelect()
        
        // 启用预处理按钮
        elements.runPreprocessBtn.disabled = false
        elements.loadWaveBtn.disabled = false
        elements.randomWaveBtn.disabled = false
      }
      
      updateButtonStates()
    }
  })
  
  // 选择测试数据
  elements.selectTestBtn.addEventListener('click', async () => {
    const path = await window.electronAPI.selectTestData()
    if (path) {
      state.testDataPath = path
      elements.testDataPath.value = path
      updateButtonStates()
    }
  })
  
  // 运行预处理
  elements.runPreprocessBtn.addEventListener('click', async () => {
    if (!state.trainDataPath) return
    
    setProcessing(true)
    updateStatus('正在预处理数据...')
    elements.preprocessStatus.className = 'status-indicator running'
    elements.preprocessStatus.textContent = '处理中...'
    
    try {
      const result = await window.electronAPI.runPreprocessing(state.trainDataPath)
      if (result.success) {
        elements.preprocessStatus.className = 'status-indicator success'
        elements.preprocessStatus.textContent = '✅ 预处理完成'
        state.processedDataReady = true
      }
    } catch (error) {
      elements.preprocessStatus.className = 'status-indicator error'
      elements.preprocessStatus.textContent = '❌ 预处理失败'
    }
    
    setProcessing(false)
    updateButtonStates()
    updateStatus('就绪')
  })
  
  // 运行训练
  elements.runTrainingBtn.addEventListener('click', async () => {
    setProcessing(true)
    updateStatus('正在训练模型...')
    elements.trainingStatus.className = 'status-indicator running'
    elements.trainingStatus.textContent = '训练中...'
    
    try {
      const result = await window.electronAPI.runTraining()
      if (result.success) {
        elements.trainingStatus.className = 'status-indicator success'
        elements.trainingStatus.textContent = '✅ 训练完成'
        elements.modelInfo.innerHTML = '<div>模型已保存，可用于预测</div>'
        state.modelReady = true
      }
    } catch (error) {
      elements.trainingStatus.className = 'status-indicator error'
      elements.trainingStatus.textContent = '❌ 训练失败'
    }
    
    setProcessing(false)
    updateButtonStates()
    updateStatus('就绪')
  })
  
  // 运行测试
  elements.runTestingBtn.addEventListener('click', async () => {
    if (!state.testDataPath) return
    
    setProcessing(true)
    updateStatus('正在进行预测...')
    elements.testingStatus.className = 'status-indicator running'
    elements.testingStatus.textContent = '预测中...'
    
    try {
      const result = await window.electronAPI.runTesting(state.testDataPath)
      if (result.success) {
        elements.testingStatus.className = 'status-indicator success'
        elements.testingStatus.textContent = '✅ 预测完成'
        
        // 加载结果
        await loadResults()
      }
    } catch (error) {
      elements.testingStatus.className = 'status-indicator error'
      elements.testingStatus.textContent = '❌ 预测失败'
    }
    
    setProcessing(false)
    updateButtonStates()
    updateStatus('就绪')
  })
  
  // 停止进程
  elements.stopBtn.addEventListener('click', async () => {
    await window.electronAPI.stopProcess()
    setProcessing(false)
    updateStatus('已停止')
  })
  
  // 加载波形
  elements.loadWaveBtn.addEventListener('click', async () => {
    const selectedFile = elements.eegFileSelect.value
    if (!selectedFile) return
    
    await loadWaveform(selectedFile)
  })
  
  // 随机选择波形
  elements.randomWaveBtn.addEventListener('click', async () => {
    if (state.eegFiles.length === 0) return
    
    const randomIndex = Math.floor(Math.random() * state.eegFiles.length)
    const randomFile = state.eegFiles[randomIndex]
    
    elements.eegFileSelect.value = randomFile
    await loadWaveform(randomFile)
  })
  
  // 清空日志
  elements.clearLogBtn.addEventListener('click', () => {
    elements.logContainer.innerHTML = ''
  })
  
  // 导出日志
  elements.exportLogBtn.addEventListener('click', () => {
    const logText = elements.logContainer.innerText
    const blob = new Blob([logText], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `sleep_analysis_log_${Date.now()}.txt`
    a.click()
    URL.revokeObjectURL(url)
  })
  
  // 标签页切换
  elements.tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const tabId = tab.getAttribute('data-tab')
      
      // 更新标签页状态
      elements.tabs.forEach(t => t.classList.remove('active'))
      tab.classList.add('active')
      
      // 更新内容显示
      elements.tabContents.forEach(content => {
        content.classList.remove('active')
        if (content.id === `${tabId}-tab`) {
          content.classList.add('active')
        }
      })
    })
  })
}

// 更新按钮状态
function updateButtonStates() {
  elements.runPreprocessBtn.disabled = !state.trainDataPath || state.isProcessing
  elements.runTrainingBtn.disabled = !state.processedDataReady || state.isProcessing
  elements.runTestingBtn.disabled = !state.modelReady || !state.testDataPath || state.isProcessing
  elements.stopBtn.disabled = !state.isProcessing
}

// 设置处理状态
function setProcessing(processing: boolean) {
  state.isProcessing = processing
  updateButtonStates()
}

// 更新状态文本
function updateStatus(text: string) {
  elements.statusText.textContent = text
}

// 更新脑电文件下拉列表
function updateEEGFileSelect() {
  elements.eegFileSelect.innerHTML = '<option value="">-- 选择脑电文件 --</option>'
  
  state.eegFiles.forEach(file => {
    const option = document.createElement('option')
    option.value = file
    option.textContent = file
    elements.eegFileSelect.appendChild(option)
  })
}

// 加载波形
async function loadWaveform(filename: string) {
  // 使用 path.join 兼容 Windows 路径
  const filePath = state.trainDataPath + '\\' + filename
  
  // 随机选择30秒片段
  const startSec = Math.floor(Math.random() * 100) * 30
  const durationSec = 30
  
  try {
    const result = await window.electronAPI.readEEGWave(filePath, startSec, durationSec)
    
    if (result.success && result.data && result.data.length > 0) {
      drawWaveform(result.data, result.startTime || startSec, result.duration || durationSec, result.sampleRate || 100)
      
      elements.waveformInfo.innerHTML = `
        <div><strong>文件:</strong> ${filename}</div>
        <div><strong>起始时间:</strong> ${(result.startTime || startSec).toFixed(1)} 秒</div>
        <div><strong>持续时间:</strong> ${result.duration || durationSec} 秒</div>
        <div><strong>采样率:</strong> ${result.sampleRate || 100} Hz</div>
        <div><strong>数据点数:</strong> ${result.data.length}</div>
      `
    } else {
      elements.waveformInfo.innerHTML = `<div class="error">加载失败: ${result.error || '无数据'}</div>`
    }
  } catch (error) {
    elements.waveformInfo.innerHTML = `<div class="error">加载失败: ${error}</div>`
  }
}

// 绘制波形图
function drawWaveform(data: number[], startTime: number, duration: number, sampleRate: number) {
  const canvas = document.getElementById('waveformChart') as HTMLCanvasElement
  const ctx = canvas.getContext('2d')
  
  if (!ctx) {
    console.error('无法获取 Canvas 上下文')
    return
  }
  
  // 检查数据有效性
  if (!data || data.length === 0) {
    elements.waveformInfo.innerHTML = '<div class="error">无有效数据</div>'
    return
  }
  
  // 销毁旧图表
  if (state.waveformChart) {
    try {
      state.waveformChart.destroy()
    } catch (e) {
      console.warn('销毁旧图表失败:', e)
    }
  }
  
  // 降采样以提高性能（每10个点取1个）
  const downsampleRate = 10
  const downsampledData: number[] = []
  const downsampledLabels: string[] = []
  
  for (let i = 0; i < data.length; i += downsampleRate) {
    downsampledData.push(data[i])
    downsampledLabels.push((startTime + i / sampleRate).toFixed(2))
  }
  
  try {
    state.waveformChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: downsampledLabels,
        datasets: [{
          label: '脑电信号 (μV)',
          data: downsampledData,
          borderColor: '#000000',
          backgroundColor: 'rgba(0, 0, 0, 0.1)',
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'top'
          },
          title: {
            display: true,
            text: '脑电信号波形图',
            font: {
              size: 16,
              weight: 'bold'
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: '时间 (秒)'
            },
            ticks: {
              maxTicksLimit: 10
            }
          },
          y: {
            title: {
              display: true,
              text: '幅值 (μV)'
            }
          }
        },
        animation: {
          duration: 0
        }
      }
    })
  } catch (error) {
    console.error('绘制波形图失败:', error)
    elements.waveformInfo.innerHTML = `<div class="error">绘制失败: ${error}</div>`
  }
}

// 加载结果
async function loadResults() {
  // 先加载预测结果图片（优先展示）
  await loadResultImages()
  
  // 再加载 CSV 结果数据
  const result = await window.electronAPI.readResults()
  
  if (result.success && Object.keys(result.results).length > 0) {
    let html = '<div class="results-list">'
    html += '<h3>预测数据详情</h3>'
    
    for (const [filename, content] of Object.entries(result.results)) {
      html += `
        <div class="result-item">
          <h4>${filename}</h4>
          <pre>${content}</pre>
        </div>
      `
    }
    
    html += '</div>'
    
    // 将 CSV 数据追加到结果容器（图片已经在上面了）
    elements.resultsContainer.innerHTML += html
  }
}

// 加载预测结果图片
async function loadResultImages() {
  try {
    // 先清空结果容器
    elements.resultsContainer.innerHTML = ''
    
    const result = await window.electronAPI.getResultImages()
    
    if (result.success && result.images.length > 0) {
      let html = '<div class="images-grid">'
      html += '<h3>📊 可视化结果</h3>'
      
      for (const img of result.images) {
        // 使用 file:// 协议加载本地图片
        const imgSrc = `file:///${img.path.replace(/\\/g, '/')}`
        html += `
          <div class="image-item">
            <h4>${img.name}</h4>
            <img src="${imgSrc}" alt="${img.name}" style="max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px;" />
          </div>
        `
      }
      
      html += '</div>'
      
      // 图片放在最前面
      elements.resultsContainer.innerHTML = html
    } else {
      // 如果没有图片，显示提示
      elements.resultsContainer.innerHTML = '<div class="no-results"><p>暂无可视化结果</p></div>'
    }
  } catch (error) {
    console.error('加载图片失败:', error)
    elements.resultsContainer.innerHTML = '<div class="no-results"><p>加载图片失败</p></div>'
  }
}

// 追加日志
function appendLog(log: string) {
  const line = document.createElement('div')
  line.className = 'log-line'
  
  // 添加时间戳
  const timestamp = new Date().toLocaleTimeString()
  line.textContent = `[${timestamp}] ${log}`
  
  // 根据内容设置样式
  if (log.includes('ERROR') || log.includes('❌') || log.includes('失败')) {
    line.classList.add('error')
  } else if (log.includes('✅') || log.includes('完成') || log.includes('成功')) {
    line.classList.add('success')
  } else if (log.includes('🔄') || log.includes('开始')) {
    line.classList.add('info')
  }
  
  elements.logContainer.appendChild(line)
  
  // 滚动到底部
  elements.logContainer.scrollTop = elements.logContainer.scrollHeight
}

// 启动应用
init()
