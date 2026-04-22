import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import LessonPage from './pages/LessonPage'
import PPTEditorPage from './pages/PPTEditorPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/lesson" replace />} />
          <Route path="lesson" element={<LessonPage />} />
          <Route path="ppt-editor" element={<PPTEditorPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
