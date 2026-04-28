import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import LessonPage from './pages/LessonPage'
import PPTEditorPage from './pages/PPTEditorPage'
import TemplateListPage from './pages/TemplateListPage'
import TemplateEditorPage from './pages/TemplateEditorPage'
import CoCreateLessonPage from './pages/CoCreateLessonPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/lesson" replace />} />
          <Route path="lesson" element={<LessonPage />} />
          <Route path="ppt-editor" element={<PPTEditorPage />} />
          <Route path="templates" element={<TemplateListPage />} />
          {/* Use * to match paths with slashes (subdirectories) */}
          <Route path="templates/edit/*" element={<TemplateEditorPage />} />
          <Route path="templates/co-create" element={<CoCreateLessonPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
