import { app, BrowserWindow, ipcMain, dialog } from 'electron'
import { join } from 'path'
import { spawn, ChildProcess, execSync } from 'child_process'
import * as fs from 'fs'
import * as path from 'path'

let win: BrowserWindow | null = null
let pythonProcess: ChildProcess | null = null

// 数据目录
const getDataDir = () => {
  const dataDir = join(app.getPath('userData'), 'data')
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true })
  }
  return dataDir
}

// 检查 Python 是否可用
function checkPython(): { available: boolean; version: string; cmd: string } {
  const pythonCmds = process.platform === 'win32' ? ['python', 'python3'] : ['python3', 'python']
  
  for (const cmd of pythonCmds) {
    try {
      const version = execSync(`${cmd} --version`, { encoding: 'utf-8', timeout: 5000 }).trim()
      return { available: true, version, cmd }
    } catch (e) {
      continue
    }
  }
  
  return { available: false, version: '', cmd: '' }
}

// 创建窗口
function createWindow() {
  win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 800,
    backgroundColor: '#ffffff',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: join(__dirname, '../preload/index.js')
    }
  })

  // 加载渲染进程页面
  if (app.isPackaged) {
    win.loadFile(join(__dirname, '../renderer/index.html'))
  } else {
    win.loadURL('http://localhost:5173')
    win.webContents.openDevTools()
  }
}

// 应用启动
app.whenReady().then(createWindow)

// 应用关闭
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow()
})

// IPC: 检查 Python 环境
ipcMain.handle('check-python', async () => {
  return checkPython()
})

// IPC: 选择训练数据目录
ipcMain.handle('select-train-data', async () => {
  const result = await dialog.showOpenDialog(win!, {
    properties: ['openDirectory'],
    title: '选择训练数据目录'
  })
  
  if (result.canceled || result.filePaths.length === 0) {
    return null
  }
  
  return result.filePaths[0]
})

// IPC: 选择测试数据目录
ipcMain.handle('select-test-data', async () => {
  const result = await dialog.showOpenDialog(win!, {
    properties: ['openDirectory'],
    title: '选择测试数据目录'
  })
  
  if (result.canceled || result.filePaths.length === 0) {
    return null
  }
  
  return result.filePaths[0]
})

// IPC: 获取数据文件列表
ipcMain.handle('get-data-files', async (_event, dataPath: string) => {
  try {
    const files = fs.readdirSync(dataPath)
    const eegFiles = files.filter(f => f.includes('EEGFpz_Cz') && f.endsWith('.txt') && !f.includes('filtered'))
    const labelFiles = files.filter(f => f.includes('Hypnogram') && f.endsWith('.txt'))
    
    return {
      eegFiles,
      labelFiles,
      total: files.length
    }
  } catch (error) {
    console.error('读取数据目录失败:', error)
    return null
  }
})

// IPC: 运行数据预处理
ipcMain.handle('run-preprocessing', async (_event, trainDataPath: string) => {
  const dataDir = getDataDir()
  const processedDir = join(dataDir, 'processed_data')
  
  if (!fs.existsSync(processedDir)) {
    fs.mkdirSync(processedDir, { recursive: true })
  }
  
  // 检查 Python
  const pythonCheck = checkPython()
  if (!pythonCheck.available) {
    win?.webContents.send('python-log', '❌ 错误：未找到 Python，请确保已安装 Python 3.8+')
    return { success: false, output: 'Python not found' }
  }
  
  win?.webContents.send('python-log', `✅ 检测到 Python: ${pythonCheck.version}`)
  
  // 读取 Python 脚本
  const scriptPath = app.isPackaged 
    ? join(process.resourcesPath, 'python', 'data_preprocessing.py')
    : join(__dirname, '../../python', 'data_preprocessing.py')
  
  // 检查脚本文件是否存在
  if (!fs.existsSync(scriptPath)) {
    win?.webContents.send('python-log', `❌ 错误：脚本文件不存在: ${scriptPath}`)
    return { success: false, output: 'Script not found' }
  }
  
  return new Promise((resolve, reject) => {
    // 发送开始消息
    win?.webContents.send('python-log', '开始数据预处理...')
    win?.webContents.send('python-log', `训练数据路径: ${trainDataPath}`)
    win?.webContents.send('python-log', `处理结果保存路径: ${processedDir}`)
    win?.webContents.send('python-log', `Python 命令: ${pythonCheck.cmd}`)
    win?.webContents.send('python-log', `脚本路径: ${scriptPath}`)
    
    const env = {
      ...process.env,
      TRAIN_DATA_PATH: trainDataPath,
      PROCESSED_DATA_PATH: processedDir
    }
    
    pythonProcess = spawn(pythonCheck.cmd, [scriptPath], { env })
    
    let output = ''
    
    pythonProcess.stdout?.on('data', (data) => {
      const text = data.toString()
      output += text
      // 按行分割并发送
      const lines = text.split('\n')
      lines.forEach((line: string) => {
        if (line.trim()) {
          win?.webContents.send('python-log', line)
        }
      })
    })
    
    pythonProcess.stderr?.on('data', (data) => {
      const text = data.toString()
      output += text
      // 按行分割并发送
      const lines = text.split('\n')
      lines.forEach((line: string) => {
        if (line.trim()) {
          win?.webContents.send('python-log', `[STDERR] ${line}`)
        }
      })
    })
    
    pythonProcess.on('error', (error) => {
      win?.webContents.send('python-log', `❌ 进程错误: ${error.message}`)
      reject({ success: false, output: error.message })
    })
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        win?.webContents.send('python-log', '✅ 数据预处理完成！')
        resolve({ success: true, output })
      } else {
        win?.webContents.send('python-log', `❌ 数据预处理失败，退出码: ${code}`)
        reject({ success: false, output, code })
      }
    })
  })
})

// IPC: 运行模型训练
ipcMain.handle('run-training', async () => {
  const dataDir = getDataDir()
  const processedDir = join(dataDir, 'processed_data')
  
  // 检查 Python
  const pythonCheck = checkPython()
  if (!pythonCheck.available) {
    win?.webContents.send('python-log', '❌ 错误：未找到 Python')
    return { success: false, output: 'Python not found' }
  }
  
  const scriptPath = app.isPackaged 
    ? join(process.resourcesPath, 'python', 'model_training.py')
    : join(__dirname, '../../python', 'model_training.py')
  
  // 检查脚本文件是否存在
  if (!fs.existsSync(scriptPath)) {
    win?.webContents.send('python-log', `❌ 错误：脚本文件不存在: ${scriptPath}`)
    return { success: false, output: 'Script not found' }
  }
  
  return new Promise((resolve, reject) => {
    win?.webContents.send('python-log', '开始模型训练...')
    win?.webContents.send('python-log', `处理数据路径: ${processedDir}`)
    
    const env = {
      ...process.env,
      PROCESSED_DATA_PATH: processedDir
    }
    
    pythonProcess = spawn(pythonCheck.cmd, [scriptPath], { env })
    
    let output = ''
    
    pythonProcess.stdout?.on('data', (data) => {
      const text = data.toString()
      output += text
      // 按行分割并发送
      const lines = text.split('\n')
      lines.forEach((line: string) => {
        if (line.trim()) {
          win?.webContents.send('python-log', line)
        }
      })
    })
    
    pythonProcess.stderr?.on('data', (data) => {
      const text = data.toString()
      output += text
      // 按行分割并发送
      const lines = text.split('\n')
      lines.forEach((line: string) => {
        if (line.trim()) {
          win?.webContents.send('python-log', `[STDERR] ${line}`)
        }
      })
    })
    
    pythonProcess.on('error', (error) => {
      win?.webContents.send('python-log', `❌ 进程错误: ${error.message}`)
      reject({ success: false, output: error.message })
    })
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        win?.webContents.send('python-log', '✅ 模型训练完成！')
        resolve({ success: true, output })
      } else {
        win?.webContents.send('python-log', `❌ 模型训练失败，退出码: ${code}`)
        reject({ success: false, output, code })
      }
    })
  })
})

// IPC: 运行模型测试
ipcMain.handle('run-testing', async (_event, testDataPath: string) => {
  const dataDir = getDataDir()
  const processedDir = join(dataDir, 'processed_data')
  const resultsDir = join(dataDir, 'results')
  
  if (!fs.existsSync(resultsDir)) {
    fs.mkdirSync(resultsDir, { recursive: true })
  }
  
  // 检查 Python
  const pythonCheck = checkPython()
  if (!pythonCheck.available) {
    win?.webContents.send('python-log', '❌ 错误：未找到 Python')
    return { success: false, output: 'Python not found' }
  }
  
  const scriptPath = app.isPackaged 
    ? join(process.resourcesPath, 'python', 'model_test.py')
    : join(__dirname, '../../python', 'model_test.py')
  
  // 检查脚本文件是否存在
  if (!fs.existsSync(scriptPath)) {
    win?.webContents.send('python-log', `❌ 错误：脚本文件不存在: ${scriptPath}`)
    return { success: false, output: 'Script not found' }
  }
  
  return new Promise((resolve, reject) => {
    win?.webContents.send('python-log', '开始模型预测...')
    win?.webContents.send('python-log', `测试数据路径: ${testDataPath}`)
    win?.webContents.send('python-log', `结果保存路径: ${resultsDir}`)
    
    const env = {
      ...process.env,
      TEST_DATA_PATH: testDataPath,
      PROCESSED_DATA_PATH: processedDir,
      RESULTS_PATH: resultsDir
    }
    
    pythonProcess = spawn(pythonCheck.cmd, [scriptPath], { env })
    
    let output = ''
    
    pythonProcess.stdout?.on('data', (data) => {
      const text = data.toString()
      output += text
      // 按行分割并发送
      const lines = text.split('\n')
      lines.forEach((line: string) => {
        if (line.trim()) {
          win?.webContents.send('python-log', line)
        }
      })
    })
    
    pythonProcess.stderr?.on('data', (data) => {
      const text = data.toString()
      output += text
      // 按行分割并发送
      const lines = text.split('\n')
      lines.forEach((line: string) => {
        if (line.trim()) {
          win?.webContents.send('python-log', `[STDERR] ${line}`)
        }
      })
    })
    
    pythonProcess.on('error', (error) => {
      win?.webContents.send('python-log', `❌ 进程错误: ${error.message}`)
      reject({ success: false, output: error.message })
    })
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        win?.webContents.send('python-log', '✅ 模型预测完成！')
        resolve({ success: true, output, resultsPath: resultsDir })
      } else {
        win?.webContents.send('python-log', `❌ 模型预测失败，退出码: ${code}`)
        reject({ success: false, output, code })
      }
    })
  })
})

// IPC: 获取预测结果图片列表
ipcMain.handle('get-result-images', async () => {
  const dataDir = getDataDir()
  const resultsDir = join(dataDir, 'results')
  
  try {
    if (!fs.existsSync(resultsDir)) {
      return { success: false, images: [] }
    }
    
    const files = fs.readdirSync(resultsDir)
    const images = files
      .filter(f => f.endsWith('.png'))
      .map(f => ({
        name: f,
        path: join(resultsDir, f)
      }))
    
    return { success: true, images }
  } catch (error) {
    return { success: false, images: [] }
  }
})

// IPC: 读取波形数据
ipcMain.handle('read-eeg-wave', async (_event, filePath: string, startSec: number, durationSec: number) => {
  try {
    const fs_rate = 100
    const startIdx = Math.floor(startSec * fs_rate)
    const endIdx = Math.floor((startSec + durationSec) * fs_rate)
    
    const data = fs.readFileSync(filePath, 'utf-8')
    const lines = data.split('\n').filter(line => line.trim() !== '')
    
    const values = []
    for (let i = startIdx; i < Math.min(endIdx, lines.length); i++) {
      const val = parseFloat(lines[i])
      if (!isNaN(val)) {
        values.push(val)
      }
    }
    
    return {
      success: true,
      data: values,
      startTime: startSec,
      duration: durationSec,
      sampleRate: fs_rate
    }
  } catch (error) {
    console.error('读取波形数据失败:', error)
    return { success: false, error: String(error) }
  }
})

// IPC: 读取预测结果
ipcMain.handle('read-results', async () => {
  const dataDir = getDataDir()
  const resultsDir = join(dataDir, 'results')
  
  try {
    const files = fs.readdirSync(resultsDir)
    const results: any = {}
    
    for (const file of files) {
      if (file.endsWith('.csv')) {
        const filePath = join(resultsDir, file)
        const content = fs.readFileSync(filePath, 'utf-8')
        results[file] = content
      }
    }
    
    return { success: true, results, resultsPath: resultsDir }
  } catch (error) {
    return { success: false, error: String(error) }
  }
})

// IPC: 停止 Python 进程
ipcMain.handle('stop-process', async () => {
  if (pythonProcess) {
    pythonProcess.kill()
    pythonProcess = null
    win?.webContents.send('python-log', '进程已停止')
    return true
  }
  return false
})

// IPC: 检查模型是否存在
ipcMain.handle('check-model', async () => {
  const dataDir = getDataDir()
  const modelPath = join(dataDir, 'processed_data', 'model', 'sleep_stage_rf_model_optimized.pkl')
  return fs.existsSync(modelPath)
})

// IPC: 检查预处理数据是否存在
ipcMain.handle('check-processed-data', async () => {
  const dataDir = getDataDir()
  const framesPath = join(dataDir, 'processed_data', 'all_eeg_frames.npy')
  return fs.existsSync(framesPath)
})
