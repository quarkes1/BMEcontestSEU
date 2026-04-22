import { contextBridge, ipcRenderer } from 'electron'

// 暴露安全的 API 给渲染进程
contextBridge.exposeInMainWorld('electronAPI', {
  // 选择训练数据目录
  selectTrainData: () => ipcRenderer.invoke('select-train-data'),
  
  // 选择测试数据目录
  selectTestData: () => ipcRenderer.invoke('select-test-data'),
  
  // 获取数据文件列表
  getDataFiles: (dataPath: string) => ipcRenderer.invoke('get-data-files', dataPath),
  
  // 运行数据预处理
  runPreprocessing: (trainDataPath: string) => ipcRenderer.invoke('run-preprocessing', trainDataPath),
  
  // 运行模型训练
  runTraining: () => ipcRenderer.invoke('run-training'),
  
  // 运行模型测试
  runTesting: (testDataPath: string) => ipcRenderer.invoke('run-testing', testDataPath),
  
  // 读取脑电波形数据
  readEEGWave: (filePath: string, startSec: number, durationSec: number) => 
    ipcRenderer.invoke('read-eeg-wave', filePath, startSec, durationSec),
  
  // 读取预测结果
  readResults: () => ipcRenderer.invoke('read-results'),
  
  // 获取预测结果图片列表
  getResultImages: () => ipcRenderer.invoke('get-result-images'),
  
  // 停止进程
  stopProcess: () => ipcRenderer.invoke('stop-process'),
  
  // 检查模型是否存在
  checkModel: () => ipcRenderer.invoke('check-model'),
  
  // 检查预处理数据是否存在
  checkProcessedData: () => ipcRenderer.invoke('check-processed-data'),
  
  // 监听 Python 日志
  onPythonLog: (callback: (log: string) => void) => {
    ipcRenderer.on('python-log', (_event, log) => callback(log))
  },
  
  // 移除监听器
  removePythonLogListener: () => {
    ipcRenderer.removeAllListeners('python-log')
  }
})
