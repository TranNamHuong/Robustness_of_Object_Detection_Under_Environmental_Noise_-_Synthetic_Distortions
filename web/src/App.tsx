import { createContext, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { NavLink, Route, Routes, useLocation } from 'react-router-dom';
import {
  Activity, ArrowDownToLine, ArrowRight, BarChart3, BookOpen, Camera,
  Check, CircleHelp, Database, FileImage, FlaskConical,
  Expand, Gauge, ImagePlus, Info, Layers3, Menu, Microscope, ScanSearch,
  ShieldCheck, Sparkles, Sun, Moon, Upload, X,
} from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api, Experiment, formatMetric, formatNumber, GalleryResponse, HealthResponse, MetricsResponse, RecommendationResponse } from './api';

const navigation = [
  { to: '/', label: 'Detection Lab', vi: 'Phát hiện vật thể', icon: ScanSearch, end: true },
  { to: '/comparison', label: 'Comparison', vi: 'So sánh & đề xuất', icon: BarChart3 },
  { to: '/analytics', label: 'Research analytics', vi: 'Phân tích nghiên cứu', icon: Activity },
  { to: '/methodology', label: 'Methodology & gallery', vi: 'Phương pháp & thư viện', icon: BookOpen },
];
type Language = 'vi' | 'en';
const LanguageContext = createContext<Language>('vi');
const useLanguage = () => useContext(LanguageContext);
const copy = (language: Language, vi: string, en: string) => language === 'vi' ? vi : en;

function App() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [theme, setTheme] = useState<'light' | 'dark'>(() => window.localStorage.getItem('robustness-lab-theme') === 'dark' ? 'dark' : 'light');
  const [language, setLanguage] = useState<Language>(() => window.localStorage.getItem('robustness-lab-language') === 'en' ? 'en' : 'vi');
  const [helpOpen, setHelpOpen] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState('');
  const location = useLocation();
  const current = navigation.find((item) => item.to === location.pathname) ?? navigation[0];
  const PageIcon = current.icon;

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem('robustness-lab-theme', theme);
  }, [theme]);
  useEffect(() => {
    window.localStorage.setItem('robustness-lab-language', language);
    document.documentElement.lang = language;
  }, [language]);

  useEffect(() => {
    if (location.pathname !== '/methodology' || !location.hash) return;
    const targetId = decodeURIComponent(location.hash.slice(1));
    // Let the route render and apply its layout before measuring the target.
    const timer = window.setTimeout(() => {
      const target = document.getElementById(targetId);
      if (!target) return;
      const top = Math.max(0, window.scrollY + target.getBoundingClientRect().top - 82);
      window.scrollTo({ top, behavior: 'smooth' });
    }, 80);
    return () => window.clearTimeout(timer);
  }, [location.pathname, location.hash]);

  useEffect(() => {
    let active = true;
    const check = () => api.health().then((value) => {
      if (active) { setHealth(value); setHealthError(''); }
    }).catch((error: Error) => {
      if (active) { setHealth(null); setHealthError(error.message); }
    });
    void check();
    const timer = window.setInterval(check, 12000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  return <LanguageContext.Provider value={language}>
    <div className="app-shell">
      {mobileOpen && <button aria-label={copy(language, 'Đóng điều hướng', 'Close navigation')} className="scrim" onClick={() => setMobileOpen(false)} />}
      <aside className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`}>
        <div className="brand-row">
          <div className="brand-mark"><Microscope size={22} strokeWidth={2.3} /></div>
          <div className="brand-text"><strong>Robust Vision</strong><span>RESEARCH LAB</span></div>
          <button aria-label={copy(language, 'Đóng menu', 'Close menu')} className="icon-button sidebar-close" onClick={() => setMobileOpen(false)}><X size={18} /></button>
        </div>

        <div className="workspace-switcher">
          <div className="workspace-icon"><FlaskConical size={17} /></div>
          <div><strong>{copy(language, 'Phòng lab nhận diện', 'Detection Lab')}</strong><span>{copy(language, 'Không gian cục bộ', 'Local workspace')}</span></div>
        </div>

        <div className="nav-caption">{copy(language, 'PHÒNG THÍ NGHIỆM', 'LAB')}</div>
        <nav aria-label={copy(language, 'Điều hướng chính', 'Main navigation')} className="main-nav">
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink key={item.to} to={item.to} end={item.end} onClick={() => setMobileOpen(false)} className={({ isActive }) => `nav-link ${isActive ? 'nav-active' : ''}`}>
                <Icon size={18} strokeWidth={1.9} />
                <span><strong>{language === 'vi' ? item.vi : item.label}</strong></span>
              </NavLink>
            );
          })}
        </nav>

        <div className="sidebar-divider" />
        <div className="nav-caption">{copy(language, 'TÀI NGUYÊN', 'RESOURCES')}</div>
          <NavLink className="resource-link" to="/methodology#provenance">{copy(language, 'Dữ liệu & nguồn', 'Datasets & sources')}</NavLink>
        <NavLink className="resource-link" to="/methodology#glossary">{copy(language, 'Thuật ngữ', 'Glossary')}</NavLink>

        <div className="sidebar-spacer" />
        <div className="side-status">
          <span className={`status-pulse ${health ? 'status-connected' : ''}`} />
          <div><strong>{copy(language, 'Chế độ nghiên cứu cục bộ', 'Local research mode')}</strong><small>{health ? `${health.device} · ${health.gpu_name ?? 'CPU'}` : healthError ? copy(language, 'Máy chủ chưa kết nối', 'Backend disconnected') : copy(language, 'Đang kiểm tra…', 'Checking…')}</small></div>
          <CircleHelp size={15} className="muted-icon" aria-label={copy(language, 'Inference chạy cục bộ cùng model và GPU trong project', 'Inference runs locally with the project model and GPU')} />
        </div>
        <div className="sidebar-footer"><span>v0.1 · COCO pretrained</span><span className="footer-lang">{copy(language, 'Tiếng Việt', 'English')}</span></div>
      </aside>

      <main className="main-shell">
        <header className="topbar">
          <button aria-label={copy(language, 'Mở điều hướng', 'Open navigation')} className="icon-button mobile-menu" onClick={() => setMobileOpen(true)}><Menu size={20} /></button>
          <div className="breadcrumbs"><span>Robust Vision</span><span className="crumb-sep">/</span><PageIcon size={14} /><strong>{language === 'vi' ? current.vi : current.label}</strong></div>
          <div className="topbar-right"><span className="topbar-status"><span className={`status-pulse ${health ? 'status-connected' : ''}`} /> {copy(language, 'Backend cục bộ', 'Local backend')} <small>{health ? copy(language, 'Đã kết nối', 'Connected') : healthError ? copy(language, 'Mất kết nối', 'Disconnected') : copy(language, 'Đang kiểm tra', 'Checking')}</small></span><label className="language-switch"><span className="sr-only">Ngôn ngữ · Language</span><select value={language} onChange={(event) => setLanguage(event.target.value as Language)} aria-label="Ngôn ngữ giao diện · Interface language"><option value="vi">Tiếng Việt</option><option value="en">English</option></select></label><button className="theme-toggle" onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')} aria-label={copy(language, theme === 'light' ? 'Bật giao diện tối' : 'Bật giao diện sáng', theme === 'light' ? 'Enable dark theme' : 'Enable light theme')} title={copy(language, theme === 'light' ? 'Giao diện tối' : 'Giao diện sáng', theme === 'light' ? 'Dark theme' : 'Light theme')}>{theme === 'light' ? <Moon size={17} /> : <Sun size={17} />}<span>{copy(language, theme === 'light' ? 'Tối' : 'Sáng', theme === 'light' ? 'Dark' : 'Light')}</span></button><button className="help-button" onClick={() => setHelpOpen(true)} title={copy(language, 'Hướng dẫn', 'Help')} aria-label={copy(language, 'Mở hướng dẫn', 'Open help')}><CircleHelp size={17} /><span>{copy(language, 'Hướng dẫn', 'Help')}</span></button><div className="avatar">RV</div></div>
        </header>
        <div className="content-wrap">
          <Routes>
            <Route path="/" element={<DetectionPage health={health} healthError={healthError} onHelp={() => setHelpOpen(true)} />} />
            <Route path="/comparison" element={<ComparisonPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/methodology" element={<MethodologyPage />} />
            <Route path="*" element={<DetectionPage health={health} healthError={healthError} onHelp={() => setHelpOpen(true)} />} />
          </Routes>
        </div>
      </main>
      {helpOpen && <GuideModal onClose={() => setHelpOpen(false)} />}
    </div>
  </LanguageContext.Provider>;
}

function PageHeading({ eyebrow, title, subtitle, action }: { eyebrow: string; title: string; subtitle: string; action?: React.ReactNode }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>{action && <div className="heading-action">{action}</div>}</div>;
}

function GuideModal({ onClose }: { onClose: () => void }) {
  const language = useLanguage();
  const steps = language === 'vi' ? [
    ['Chọn ảnh', 'Tải ảnh JPG, PNG hoặc WebP, hoặc chụp bằng webcam. Ảnh được xử lý cục bộ và không lưu vào server.'],
    ['Chọn model và điều kiện ảnh', 'YOLOv8n nhẹ hơn; YOLOv8s thường chính xác hơn nhưng chậm hơn. Chọn Ảnh gốc hoặc một điều kiện nhiễu mô phỏng.'],
    ['Chọn cường độ và khôi phục', 'Chỉ các mức đã đo mới được chọn. Auto dùng phương pháp khôi phục được cấu hình cho mức đó.'],
    ['Điều chỉnh ngưỡng rồi chạy', 'Confidence lọc dự đoán yếu; NMS IoU lọc hộp trùng. Bấm Phát hiện hoặc So sánh model.'],
    ['Đọc kết quả', 'Ảnh tải lên không có nhãn chỉ có boxes và latency, không có mAP/TP/FP/FN. Dùng trang So sánh và Phân tích để xem benchmark có nhãn.'],
  ] : [
    ['Choose an image', 'Upload a JPG, PNG, or WebP image, or capture a webcam frame. Images are processed locally and are not stored.'],
    ['Choose a model and image condition', 'YOLOv8n is lighter; YOLOv8s is usually more accurate but slower. Choose Clean or a simulated distortion.'],
    ['Set severity and restoration', 'Only measured severity levels are available. Auto uses the configured restoration method for that level.'],
    ['Adjust thresholds and run', 'Confidence filters weak detections; NMS IoU suppresses duplicate boxes. Select Detect or Compare models.'],
    ['Read the results', 'Unlabeled uploads show boxes and latency only, not mAP/TP/FP/FN. Use Comparison and Analytics for labeled benchmark results.'],
  ];
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  return <div className="guide-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}><section className="guide-modal panel" role="dialog" aria-modal="true" aria-labelledby="guide-title"><div className="panel-heading"><div><div className="section-kicker">{copy(language, 'HƯỚNG DẪN NHANH', 'QUICK START')}</div><h2 id="guide-title">{copy(language, 'Cách dùng Detection Lab', 'How to use the lab')}</h2></div><button className="icon-button" onClick={onClose} aria-label={copy(language, 'Đóng hướng dẫn', 'Close help')}><X size={18} /></button></div>
    <ol className="guide-steps">{steps.map(([title, body]) => <li key={title}><strong>{title}</strong><p>{body}</p></li>)}</ol>
    <div className="guide-footer"><span>{copy(language, 'Chạy cục bộ', 'Runs locally')}</span><button className="button button-primary" onClick={onClose}>{copy(language, 'Đã hiểu', 'Got it')}</button></div></section></div>;
}

function StateNotice({ icon: Icon = Info, title, children, kind = 'neutral' }: { icon?: typeof Info; title: string; children: React.ReactNode; kind?: string }) {
  return <div className={`state-notice state-${kind}`}><div className="notice-icon"><Icon size={17} /></div><div><strong>{title}</strong><p>{children}</p></div></div>;
}

const CONDITION_OPTIONS = [
  { id: 'clean', vi: 'Ảnh gốc', en: 'Clean image', levels: [] },
  { id: 'gaussian', vi: 'Nhiễu Gaussian', en: 'Gaussian noise', levels: [0.01, 0.03, 0.06] },
  { id: 'gaussian_nlm', vi: 'Gaussian + NLM · Khử nhiễu', en: 'Gaussian + NLM denoising', levels: [0.01, 0.03, 0.06] },
  { id: 'salt_pepper', vi: 'Nhiễu muối tiêu', en: 'Salt-and-pepper noise', levels: [0.01, 0.03, 0.06] },
  { id: 'fog', vi: 'Sương mù mô phỏng', en: 'Synthetic fog', levels: [0.15, 0.3, 0.5] },
  { id: 'low_light', vi: 'Thiếu sáng', en: 'Low light', levels: [0.25, 0.4, 0.6] },
];

function DetectionPage({ health, healthError, onHelp }: { health: HealthResponse | null; healthError: string; onHelp: () => void }) {
  const language = useLanguage();
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [model, setModel] = useState('yolov8s');
  const [condition, setCondition] = useState('clean');
  const [severity, setSeverity] = useState(0.06);
  const [confidence, setConfidence] = useState(0.25);
  const [nmsIou, setNmsIou] = useState(0.7);
  const [recovery, setRecovery] = useState('auto');
  const [result, setResult] = useState<Awaited<ReturnType<typeof api.detect>> | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [cameraError, setCameraError] = useState('');
  const [stream, setStream] = useState<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const selectedCondition = CONDITION_OPTIONS.find((item) => item.id === condition)!;
  const previewUrl = useMemo(() => imageFile ? URL.createObjectURL(imageFile) : null, [imageFile]);
  const modelReady = Boolean(health?.models[model]);

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);
  useEffect(() => {
    if (videoRef.current && stream) videoRef.current.srcObject = stream;
  }, [stream]);
  useEffect(() => () => stream?.getTracks().forEach((track) => track.stop()), [stream]);

  function chooseFile(file?: File) {
    setError('');
    if (!file) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) { setError('Chỉ nhận file JPEG, PNG hoặc WebP.'); return; }
    if (file.size > 20 * 1024 * 1024) { setError('Ảnh vượt giới hạn upload 20 MB.'); return; }
    setImageFile(file);
    setResult(null);
  }

  function changeCondition(next: string) {
    setCondition(next);
    const levels = CONDITION_OPTIONS.find((item) => item.id === next)?.levels ?? [];
    setSeverity((current) => next === 'clean' ? 0.06 : levels.includes(current) ? current : levels[0] ?? current);
    setResult(null);
  }

  async function startCamera() {
    setCameraError('');
    if (!navigator.mediaDevices?.getUserMedia) { setCameraError('Trình duyệt này không hỗ trợ webcam hoặc trang chưa chạy trên localhost/HTTPS.'); return; }
    try { setStream(await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false })); }
    catch (cause) { setCameraError(cause instanceof Error ? `Không mở được webcam: ${cause.message}` : 'Không mở được webcam. Kiểm tra quyền camera của trình duyệt.'); }
  }

  function stopCamera() { stream?.getTracks().forEach((track) => track.stop()); setStream(null); }

  async function captureFrame() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) { setCameraError('Webcam chưa sẵn sàng.'); return; }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext('2d')?.drawImage(video, 0, 0);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92));
    if (!blob) { setCameraError('Không chụp được frame từ webcam.'); return; }
    chooseFile(new File([blob], `webcam-${Date.now()}.jpg`, { type: 'image/jpeg' }));
    stopCamera();
  }

  async function run(compareAll: boolean) {
    if (!imageFile) return;
    setBusy(true); setError(''); setResult(null);
    const form = new FormData();
    form.append('file', imageFile);
    if (compareAll) ['yolov8n', 'yolov8s', 'ssdlite'].forEach((id) => { if (health?.models[id]) form.append('models', id); });
    else form.append('model', model);
    form.append('condition', condition);
    form.append('severity', String(condition === 'clean' ? 0 : severity));
    form.append('confidence', String(confidence));
    form.append('nms_iou', String(nmsIou));
    form.append('seed', '42');
    form.append('recovery', condition === 'gaussian_nlm' ? 'nlmeans' : recovery);
    try { setResult(compareAll ? await api.compare(form) : await api.detect(form)); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Inference thất bại.'); }
    finally { setBusy(false); }
  }

  const outputModels = result ? Object.entries(result.models) : [];
  const primary = result?.models[model] ?? outputModels[0]?.[1];
  const resultViews = primary?.views;

  function downloadJson() {
    if (!result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' }));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'robustness-detection.json'; anchor.click(); URL.revokeObjectURL(url);
  }

  return <>
    <PageHeading eyebrow={copy(language, 'SUY LUẬN TRỰC TIẾP', 'LIVE INFERENCE')} title={copy(language, 'Phát hiện vật thể', 'Object detection')} subtitle={copy(language, 'Thử độ bền mô hình trên ảnh gốc, ảnh biến dạng và ảnh khôi phục.', 'Test model robustness on clean, distorted, and restored images.')} action={<button className="button button-quiet" onClick={onHelp} title={copy(language, 'Hướng dẫn sử dụng', 'User guide')}><CircleHelp size={16} /> {copy(language, 'Hướng dẫn', 'Help')}</button>} />
    <div className="info-strip"><Sparkles size={16} /><span>{copy(language, 'Inference dùng model pretrained COCO. Ảnh tải lên không có nhãn nên không hiển thị mAP, TP, FP hoặc FN.', 'Inference uses COCO-pretrained models. Unlabeled uploads do not show mAP, TP, FP, or FN.')}</span></div>
    {!health && <StateNotice icon={Activity} title={healthError ? copy(language, 'Backend chưa kết nối', 'Backend disconnected') : copy(language, 'Đang kiểm tra backend', 'Checking backend')} kind={healthError ? 'error' : 'loading'}>{healthError || copy(language, 'Đang kiểm tra weights và thiết bị cục bộ. Chạy run_local.ps1 để bật API.', 'Checking local model weights and device. Run run_local.ps1 to start the API.')}</StateNotice>}
    <section className="panel control-panel" aria-label={copy(language, 'Cấu hình phát hiện', 'Detection settings')}>
      <div className="panel-heading"><div><div className="section-kicker">01 / {copy(language, 'CẤU HÌNH', 'CONFIGURATION')}</div><h2>{copy(language, 'Cấu hình thử nghiệm', 'Experiment settings')}</h2></div><span className="badge badge-local"><span className={`badge-dot ${health ? 'status-connected' : ''}`} /> {copy(language, 'Chạy cục bộ', 'Local inference')} · {health?.device ?? 'offline'}</span></div>
      <div className="control-grid">
        <div className="field field-upload"><label htmlFor="image-upload">{copy(language, 'Ảnh đầu vào', 'Input image')}</label><label htmlFor="image-upload" className="upload-box" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); chooseFile(event.dataTransfer.files[0]); }}><input id="image-upload" type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => chooseFile(event.target.files?.[0])} /><span className="upload-icon"><Upload size={19} /></span><span className="upload-copy"><strong>{imageFile?.name || copy(language, 'Chọn hoặc kéo ảnh vào đây', 'Choose or drop an image here')}</strong><small>{copy(language, 'PNG, JPG hoặc WebP · tối đa 20 MB', 'PNG, JPG, or WebP · up to 20 MB')}</small></span><span className="upload-action"><ImagePlus size={17} /> {copy(language, 'Chọn ảnh', 'Browse')}</span></label></div>
        <div className="field"><label htmlFor="model-select">{copy(language, 'Mô hình phát hiện', 'Detection model')}</label><select id="model-select" value={model} onChange={(event) => setModel(event.target.value)}><option value="yolov8s">YOLOv8s · {copy(language, 'Cân bằng', 'Balanced')}</option><option value="yolov8n">YOLOv8n · {copy(language, 'Nhẹ, nhanh', 'Lightweight')}</option><option value="ssdlite" disabled={!health?.models.ssdlite}>SSD-MobileNet · COCO {!health?.models.ssdlite ? copy(language, '(thiếu weights cục bộ)', '(weights not installed)') : ''}</option></select>{!health?.models.ssdlite && <small className="field-hint">{copy(language, 'SSD chưa chạy được vì máy chưa có file trọng số pretrained.', 'SSD is unavailable because its pretrained weights are not installed locally.')}</small>}</div>
        <div className="field"><label htmlFor="condition-select">{copy(language, 'Điều kiện ảnh', 'Image condition')}</label><select id="condition-select" value={condition} onChange={(event) => changeCondition(event.target.value)} aria-label={copy(language, 'Điều kiện ảnh', 'Image condition')}>{CONDITION_OPTIONS.map((item) => <option value={item.id} key={item.id}>{item[language]}</option>)}</select></div>
        <div className="field field-range"><label htmlFor="confidence">{copy(language, 'Ngưỡng tin cậy', 'Confidence threshold')}</label><div className="range-wrap"><input id="confidence" type="range" min="0.05" max="0.95" step="0.05" value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} /><output>{confidence.toFixed(2)}</output></div><small className="field-hint">{copy(language, 'Ẩn các dự đoán có điểm thấp hơn.', 'Hide detections with lower scores.')}</small></div>
        <div className="field"><label htmlFor="iou">{copy(language, 'Ngưỡng loại hộp trùng · NMS IoU', 'NMS IoU threshold')}</label><div className="input-suffix"><input id="iou" type="number" min="0.1" max="0.95" step="0.05" value={nmsIou} onChange={(event) => setNmsIou(Number(event.target.value))} /><span>IoU</span></div><small className="field-hint">{copy(language, 'Loại hộp chồng lấp vượt ngưỡng này.', 'Suppress overlapping boxes above this value.')}</small></div>
        {condition !== 'clean' && <div className="field"><label htmlFor="severity">{copy(language, 'Cường độ nhiễu', 'Noise severity')}</label><select id="severity" value={severity} onChange={(event) => setSeverity(Number(event.target.value))}>{selectedCondition.levels.map((level) => <option value={level} key={level}>{level.toFixed(2)}</option>)}</select><small className="field-hint">{copy(language, 'Chỉ hiện các mức đã được đo.', 'Only measured levels are available.')}</small></div>}
        {condition !== 'clean' && condition !== 'gaussian_nlm' && <div className="field"><label htmlFor="recovery">{copy(language, 'Khôi phục ảnh', 'Image restoration')}</label><select id="recovery" value={recovery} onChange={(event) => setRecovery(event.target.value)}><option value="auto">{copy(language, 'Tự chọn theo cấu hình', 'Automatic')}</option><option value="none">{copy(language, 'Không khôi phục', 'None')}</option><option value="median">{copy(language, 'Lọc trung vị', 'Median filter')}</option><option value="nlmeans">{copy(language, 'Khử nhiễu NLM', 'Non-local means')}</option><option value="clahe">{copy(language, 'Cân bằng sáng cục bộ', 'CLAHE')}</option><option value="bilateral">{copy(language, 'Lọc song phương', 'Bilateral filter')}</option></select><small className="field-hint">{copy(language, 'Bộ lọc có thể làm mất chi tiết nhỏ.', 'Filters may remove fine details.')}</small></div>}
      </div>
      {cameraError && <StateNotice icon={Camera} title={copy(language, 'Webcam không khả dụng', 'Webcam unavailable')} kind="error">{cameraError}</StateNotice>}
      {error && <StateNotice icon={Info} title={copy(language, 'Không chạy được nhận diện', 'Inference failed')} kind="error">{error}</StateNotice>}
      <div className="control-footer"><div className="seed-note"><ShieldCheck size={15} /><span>Seed <b>42</b> · {modelReady ? `${health?.device} · ${copy(language, 'checkpoint sẵn sàng', 'checkpoint ready')}` : copy(language, 'model chưa sẵn sàng', 'model not ready')}</span></div><div className="button-row"><button className="button button-quiet" onClick={() => void startCamera()} disabled={!health || busy}><Camera size={16} /> {copy(language, 'Webcam', 'Webcam')}</button><button className="button button-outline" onClick={() => void run(true)} disabled={!imageFile || busy || !health || Object.values(health.models).filter((value) => value === true).length < 2}><Layers3 size={16} /> {copy(language, 'So sánh model', 'Compare models')}</button><button className="button button-primary" onClick={() => void run(false)} disabled={!imageFile || busy || !health || !modelReady}>{busy ? <Activity className="spin" size={16} /> : <ScanSearch size={16} />}{busy ? copy(language, 'Đang xử lý…', 'Processing…') : copy(language, 'Phát hiện', 'Detect')}<ArrowRight size={15} /></button></div></div>
    </section>
    {stream && <div className="camera-backdrop" role="dialog" aria-modal="true" aria-label={copy(language, 'Chụp webcam', 'Webcam capture')}><div className="camera-modal panel"><div className="panel-heading"><div><div className="section-kicker">CAMERA · PRIVATE LOCAL CAPTURE</div><h2>{copy(language, 'Chụp một khung hình từ webcam', 'Capture a webcam frame')}</h2></div><button className="icon-button" onClick={stopCamera} aria-label={copy(language, 'Đóng webcam', 'Close webcam')}><X size={18} /></button></div><video ref={videoRef} autoPlay playsInline className="camera-video" /><p>{copy(language, 'Khung hình chỉ được gửi khi bạn chụp và chạy nhận diện. Luồng webcam không được lưu.', 'The frame is sent only when you capture and run inference. The webcam stream is not stored.')}</p><button className="button button-primary" onClick={() => void captureFrame()}><Camera size={16} /> {copy(language, 'Chụp khung hình', 'Capture frame')}</button></div></div>}
    <div className="section-header result-header"><div><div className="section-kicker">02 / {copy(language, 'KẾT QUẢ', 'RESULTS')}</div><h2>{copy(language, 'Kết quả suy luận', 'Inference results')} <span>Inference results</span></h2></div><span className="muted-label">{result ? `${outputModels.length} ${copy(language, 'model', 'models')} · ${result.image_size.width}×${result.image_size.height}` : busy ? copy(language, 'Đang chạy model…', 'Running model…') : copy(language, 'Chưa có lần chạy nào', 'No run yet')}</span></div>
    {result && <div className="result-toolbar"><div><strong>{primary?.model ?? 'Model'}</strong><span>{selectedCondition[language]} · {copy(language, 'cường độ', 'severity')} {formatNumber(result.severity, 2)} · conf {result.confidence_threshold.toFixed(2)} · NMS IoU {result.nms_iou_threshold.toFixed(2)}</span></div><div><button className="button button-outline" onClick={downloadJson}><ArrowDownToLine size={15} /> JSON</button></div></div>}
    <section className="viewer-grid" aria-label={copy(language, 'Ảnh kết quả', 'Result images')}>
      <ImagePanel title={copy(language, 'Ảnh gốc', 'Original image')} subtitle="Original input" icon={FileImage} state={imageFile ? copy(language, 'Ảnh đã sẵn sàng', 'Image ready') : copy(language, 'Chờ ảnh đầu vào', 'Waiting for an image')} src={primary?.images.original ?? previewUrl} result={resultViews?.original} downloadName="original-detections.jpg" />
      <ImagePanel title={condition === 'clean' ? copy(language, 'Ảnh đầu vào', 'Input image') : copy(language, 'Ảnh biến dạng', 'Distorted image')} subtitle={condition === 'clean' ? copy(language, 'Không áp dụng biến dạng', 'No distortion applied') : 'Corrupted input'} icon={Activity} state={copy(language, 'Chọn điều kiện để tạo ảnh', 'Choose a condition to create this view')} src={primary?.images.noisy ?? null} result={resultViews?.noisy} downloadName="distorted-detections.jpg" />
      <ImagePanel title={result?.recovery_method ? `${copy(language, 'Sau khôi phục', 'Restored image')} · ${result.recovery_method}` : copy(language, 'Sau khôi phục', 'Restored image')} subtitle="Restored image" icon={Sparkles} state={condition === 'clean' ? copy(language, 'Không cần khôi phục ảnh gốc', 'No restoration for a clean image') : copy(language, 'Không áp dụng phương pháp khôi phục', 'No restoration method applied')} src={primary?.images.restored ?? null} result={resultViews?.restored} downloadName="restored-detections.jpg" />
    </section>
    {result && <section className="panel detection-table-panel"><div className="panel-heading"><div><div className="section-kicker">{copy(language, 'KẾT QUẢ PHÁT HIỆN · KHÔNG CÓ NHÃN CHUẨN', 'DETECTIONS · NO GROUND TRUTH')}</div><h2>{copy(language, 'Hộp, lớp và độ tin cậy', 'Boxes, classes, and confidence')} <span>Per-model output</span></h2></div><span className="badge badge-local">{copy(language, 'mAP / TP / FP / FN chưa khả dụng', 'mAP / TP / FP / FN unavailable')}</span></div><div className="table-scroll"><table className="data-table"><thead><tr><th>{copy(language, 'Mô hình', 'Model')}</th><th>{copy(language, 'Ảnh', 'View')}</th><th>{copy(language, 'Lớp', 'Class')}</th><th>Confidence</th><th>{copy(language, 'Hộp giới hạn · xyxy', 'Bounding box · xyxy')}</th><th>Latency</th></tr></thead><tbody>{outputModels.flatMap(([modelId, output]) => Object.entries(output.views).flatMap(([viewName, view]) => view.detections.map((detection, index) => <tr key={`${modelId}-${viewName}-${index}`}><td>{output.model}</td><td>{viewName === 'original' ? copy(language, 'Ảnh gốc', 'Original') : viewName === 'noisy' ? copy(language, 'Ảnh biến dạng', 'Distorted') : copy(language, 'Đã khôi phục', 'Restored')}</td><td>{detection.label}</td><td>{detection.confidence.toFixed(3)}</td><td className="bbox-cell">{detection.bbox_xyxy.map((value) => Math.round(value)).join(', ')}</td><td>{formatNumber(view.latency_ms, 1)} ms</td></tr>)))}</tbody></table></div><div className="bottom-note"><Info size={15} /><span>{copy(language, 'Ảnh này không có ground truth; confidence không phải accuracy. Ảnh tải lên/webcam không có TP/FP/FN hoặc mAP.', 'This image has no ground truth; confidence is not accuracy. Uploads/webcam frames do not provide TP/FP/FN or mAP.')}</span></div></section>}
    {result && outputModels.length > 1 && <section className="panel model-summary-panel"><div className="panel-heading"><div><div className="section-kicker">{copy(language, 'CÙNG ẢNH · CÙNG CẤU HÌNH', 'SAME INPUT · SAME SETTINGS')}</div><h2>{copy(language, 'So sánh nhanh', 'Quick comparison')} <span>Compare all models</span></h2></div></div><div className="model-summary-grid">{outputModels.map(([modelId, output]) => <div className="model-summary-card" key={modelId}><strong>{output.model}</strong><span>{output.views.noisy.detections.length} {copy(language, 'phát hiện', 'detections')}</span><small>{formatNumber(output.views.noisy.latency_ms, 1)} ms · {output.views.noisy.device}</small></div>)}</div></section>}
    <div className="bottom-note"><Info size={15} /><span>{result?.provenance.upload_retention || copy(language, 'Ba khung ảnh được xử lý từ cùng một ảnh đầu vào.', 'The three views are processed from the same input image.')} {copy(language, 'TP/FP/FN chỉ tính được khi có ground truth và quy tắc matching được xác định.', 'TP/FP/FN require ground truth and a defined matching rule.')}</span></div>
  </>;
}

function ImagePanel({ title, subtitle, icon: Icon, state, src, result, downloadName }: { title: string; subtitle: string; icon: typeof FileImage; state: string; src: string | null; result?: { detections: { label: string; confidence: number }[]; latency_ms: number; device: string }; downloadName: string }) {
  const language = useLanguage();
  const [zoomOpen, setZoomOpen] = useState(false);
  useEffect(() => {
    if (!zoomOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === 'Escape') setZoomOpen(false); };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [zoomOpen]);
  return <><article className="panel image-panel"><div className="image-panel-head"><div><strong>{title}</strong><small>{subtitle}</small></div><div className="image-actions">{src && <button className="icon-button zoom-icon-button" aria-label={copy(language, `Phóng to ${title}`, `Enlarge ${title}`)} title={copy(language, 'Phóng to toàn màn hình', 'View full screen')} onClick={() => setZoomOpen(true)}><Expand size={16} /></button>}{src && <a className="icon-button" aria-label={copy(language, `Tải ${title}`, `Download ${title}`)} title={copy(language, 'Tải ảnh có bounding box', 'Download image with detections')} href={src} download={downloadName}><ArrowDownToLine size={16} /></a>}</div></div>{src ? <button className="image-zoom-trigger" onClick={() => setZoomOpen(true)} aria-label={copy(language, `Bấm để phóng to ${title}`, `Click to enlarge ${title}`)}><div className="result-image-wrap"><img src={src} alt={`${title} with detection boxes`} /></div></button> : <div className="image-placeholder"><div className="placeholder-mark"><Icon size={23} /></div><span>{state}</span><small>{copy(language, 'Nhãn lớp và confidence sẽ hiển thị tại đây', 'Class labels and confidence scores appear here')}</small></div>}<div className="image-panel-foot"><span><Gauge size={14} /> {copy(language, 'Thời gian', 'Latency')} {result ? `${formatNumber(result.latency_ms, 1)} ms` : '—'}</span><span><ScanSearch size={14} /> {result ? `${result.detections.length} ${copy(language, 'phát hiện', 'detections')}` : '—'}</span></div></article>{zoomOpen && src && <div className="image-lightbox" role="dialog" aria-modal="true" aria-label={copy(language, 'Ảnh phóng to', 'Enlarged image')} onMouseDown={(event) => { if (event.target === event.currentTarget) setZoomOpen(false); }}><button className="icon-button lightbox-close" onClick={() => setZoomOpen(false)} aria-label={copy(language, 'Đóng ảnh lớn', 'Close enlarged image')}><X size={20} /></button><img src={src} alt={`${title} enlarged with detections`} /><div className="lightbox-caption">{title} · {result ? `${result.detections.length} ${copy(language, 'hộp phát hiện', 'detections')}` : ''}</div></div>}</>;
}

function useExperimentData(defaultId: string) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [experimentId, setExperimentId] = useState(defaultId);
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    api.experiments().then((items) => {
      if (!active) return;
      setExperiments(items.filter((item) => item.available));
      if (!items.some((item) => item.id === defaultId && item.available)) {
        const first = items.find((item) => item.available);
        if (first) setExperimentId(first.id);
      }
    }).catch((cause: Error) => { if (active) setError(cause.message); });
    return () => { active = false; };
  }, [defaultId]);

  useEffect(() => {
    if (!experimentId) return;
    let active = true;
    setLoading(true); setError('');
    api.metrics(experimentId).then((value) => { if (active) setData(value); })
      .catch((cause: Error) => { if (active) setError(cause.message); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [experimentId]);

  return { experiments, experimentId, setExperimentId, data, loading, error };
}

type MetricKey = 'mAP50_95' | 'mAP50' | 'precision' | 'recall' | 'f1' | 'latency_ms';
const METRIC_OPTIONS: { id: MetricKey; vi: string; en: string }[] = [
  { id: 'mAP50_95', vi: 'mAP@0.5:0.95 · độ chính xác trung bình', en: 'mAP@0.5:0.95 · mean average precision' }, { id: 'mAP50', vi: 'mAP@0.5 · độ chính xác IoU 0,5', en: 'mAP@0.5 · precision at IoU 0.5' },
  { id: 'precision', vi: 'Precision · độ chuẩn xác', en: 'Precision' }, { id: 'recall', vi: 'Recall · độ bao phủ', en: 'Recall' },
  { id: 'f1', vi: 'F1 · cân bằng P/R', en: 'F1 · precision/recall balance' }, { id: 'latency_ms', vi: 'Latency · thời gian (ms)', en: 'Latency (ms)' },
];
const MODEL_COLORS = ['#7c3aed', '#0891b2', '#f59e0b', '#10b981', '#ec4899'];
function translateFamily(value: string, language: Language) {
  const labels: Record<string, [string, string]> = { gaussian: ['Nhiễu Gaussian', 'Gaussian noise'], salt_pepper: ['Nhiễu muối tiêu', 'Salt-and-pepper noise'], fog: ['Sương mù', 'Fog'], low_light: ['Thiếu sáng', 'Low light'], clean: ['Ảnh gốc', 'Clean'] };
  return labels[value]?.[language === 'vi' ? 0 : 1] ?? value;
}

function DatasetSelector({ experiments, value, onChange, id = 'dataset-filter' }: { experiments: Experiment[]; value: string; onChange: (value: string) => void; id?: string }) {
  const language = useLanguage();
  const vi: Record<string, string> = { coco_normalized_320: 'COCO · đánh giá chung · 320 px', coco_yolov8s_640_recovery: 'COCO · thử nghiệm khôi phục · 640 px', coco_3x500_subsets: 'COCO · 3 tập con, mỗi tập 500 ảnh', coco_yolov8_clean_640: 'COCO · so sánh ảnh sạch · 640 px', open_images_100: 'Open Images · tập ngoài 100 ảnh', open_images_300: 'Open Images · tập ngoài 300 ảnh', user_reference_unverified: 'Bảng tham khảo chưa xác minh' };
  const en: Record<string, string> = { coco_normalized_320: 'COCO · shared evaluation · 320 px', coco_yolov8s_640_recovery: 'COCO · recovery study · 640 px', coco_3x500_subsets: 'COCO · 3 subsets, 500 images each', coco_yolov8_clean_640: 'COCO · clean comparison · 640 px', open_images_100: 'Open Images · 100-image external set', open_images_300: 'Open Images · 300-image external set', user_reference_unverified: 'Unverified reference table' };
  const selected = experiments.find((item) => item.id === value);
  const groups = [
    { id: 'comparison', label: copy(language, 'So sánh model', 'Model comparison'), items: experiments.filter((item) => !item.reference_only && item.kind === 'clean_model_comparison' || !item.reference_only && item.kind === 'multi_model_corruption') },
    { id: 'robustness', label: copy(language, 'Độ bền & khôi phục', 'Robustness & recovery'), items: experiments.filter((item) => !item.reference_only && ['single_model_recovery', 'three_subset_aggregate'].includes(item.kind)) },
    { id: 'external', label: copy(language, 'Đánh giá dataset ngoài', 'External dataset evaluation'), items: experiments.filter((item) => !item.reference_only && item.kind === 'external_labeled_subset') },
    { id: 'reference', label: copy(language, 'Tài liệu tham khảo', 'Reference only'), items: experiments.filter((item) => item.reference_only) },
  ].filter((group) => group.items.length > 0);
  const purpose = (item: Experiment) => item.reference_only
    ? copy(language, 'Chưa xác minh', 'Unverified')
    : item.kind === 'multi_model_corruption' || item.kind === 'clean_model_comparison'
      ? copy(language, 'So sánh model', 'Model comparison')
      : item.kind === 'external_labeled_subset'
        ? copy(language, 'Dataset ngoài', 'External dataset')
        : copy(language, 'Phân tích một model', 'Single-model analysis');
  const metricCoverage = (item: Experiment) => [
    { label: 'mAP@0.5', count: item.map50_count ?? 0 },
    { label: 'mAP@0.5:0.95', count: item.map50_95_count ?? 0 },
  ].filter((metric) => metric.count > 0);
  return <>
    <select id={id} value={value} onChange={(event) => onChange(event.target.value)} aria-label={copy(language, 'Chọn bộ dữ liệu và lần đánh giá', 'Select dataset and evaluation')}>
      {groups.map((group) => <optgroup key={group.id} label={group.label}>
        {group.items.map((item) => {
          const title = (language === 'vi' ? vi : en)[item.id] ?? item.title;
          const count = item.reference_only
            ? copy(language, ' · chưa xác minh', ' · unverified')
            : ` · ${item.model_count ?? 0} ${copy(language, 'model', 'models')} · ${item.map50_95_count ? 'mAP95' : item.map50_count ? 'mAP50' : copy(language, 'chưa có metric', 'no metric')}`;
          return <option key={item.id} value={item.id}>{title}{count}</option>;
        })}
      </optgroup>)}
    </select>
    {selected && <div className="dataset-coverage" aria-live="polite">
      <div className="dataset-coverage-heading"><strong>{(language === 'vi' ? vi : en)[selected.id] ?? selected.title}</strong><span className={`dataset-purpose ${selected.reference_only ? 'is-reference' : ''}`}>{purpose(selected)}</span></div>
      <div className="dataset-coverage-badges">
        {!selected.reference_only && <span>{selected.record_count ?? 0} {copy(language, 'kết quả đo', 'records')}</span>}
        {!selected.reference_only && <span>{selected.model_count ?? 0} {copy(language, 'model', 'models')}</span>}
        {!selected.reference_only && <span>{selected.conditions?.length ?? 0} {copy(language, 'điều kiện', 'conditions')}</span>}
        {metricCoverage(selected).map((metric) => <span className="metric-available" key={metric.label}>{metric.label} · {metric.count}/{selected.record_count ?? 0}</span>)}
        {!selected.reference_only && metricCoverage(selected).length === 0 && <span className="metric-unavailable">{copy(language, 'Chưa có metric đo', 'No measured metric')}</span>}
        {!selected.reference_only && !selected.map50_95_count && (selected.map50_count ?? 0) > 0 && <span className="metric-unavailable">mAP@0.5:0.95 · {copy(language, 'chưa đo', 'not measured')}</span>}
        {!selected.reference_only && (selected.model_count ?? 0) < 2 && <span className="metric-unavailable">{copy(language, 'Chưa đủ model để so sánh', 'Not enough models to compare')}</span>}
      </div>
      {selected.reference_only && <small>{copy(language, 'Không dùng để xếp hạng hoặc so sánh benchmark.', 'Excluded from benchmark ranking and comparison.')}</small>}
    </div>}
  </>;
}

function ComparisonPage() {
  const language = useLanguage();
  const { experiments, experimentId, setExperimentId, data, loading, error } = useExperimentData('coco_normalized_320');
  const [condition, setCondition] = useState('clean');
  const [stage, setStage] = useState('clean');
  const [model, setModel] = useState('all');
  const [metric, setMetric] = useState<MetricKey>('mAP50_95');
  const rows = data?.records ?? [];
  const conditions = data?.filters.conditions ?? [];
  const models = data?.filters.models ?? [];
  const activeCondition = conditions.includes(condition) ? condition : conditions[0] ?? '';
  const activeStages = [...new Set(rows.filter((row) => row.condition === activeCondition).map((row) => row.stage))];
  const activeStage = stage === 'all' || activeStages.includes(stage) ? stage : activeStages[0] ?? 'all';
  const availableRows = rows.filter((row) => row.condition === activeCondition && (activeStage === 'all' || row.stage === activeStage) && (model === 'all' || row.model === model));
  const chartData = availableRows.map((row) => ({ ...row, value: row[metric] as number | null }));
  const selectedExperiment = loading && data ? data.experiment : experiments.find((item) => item.id === experimentId) ?? data?.experiment;
  const metricLabel = METRIC_OPTIONS.find((item) => item.id === metric)?.[language] ?? metric;
  const isReference = data?.reference_only ?? false;
  const [criterion, setCriterion] = useState('map');
  const recommendationStage = criterion === 'robustness' || criterion === 'recovery'
    ? undefined
    : activeStage === 'all' ? (activeCondition === 'clean' ? 'clean' : 'raw') : activeStage;
  const [recommendation, setRecommendation] = useState<RecommendationResponse | null>(null);
  useEffect(() => {
    let active = true;
    setRecommendation(null);
    api.recommendations(experimentId, criterion, activeCondition, recommendationStage).then((value) => { if (active) setRecommendation(value); }).catch(() => { if (active) setRecommendation(null); });
    return () => { active = false; };
  }, [experimentId, criterion, activeCondition, recommendationStage]);

  function changeExperiment(value: string) {
    const nextExperiment = experiments.find((item) => item.id === value);
    setExperimentId(value); setCondition('clean'); setModel('all'); setStage('clean');
    setMetric(nextExperiment?.map50_95_count ? 'mAP50_95' : nextExperiment?.map50_count ? 'mAP50' : 'mAP50_95');
  }

  return <>
    <PageHeading eyebrow={copy(language, 'BENCHMARK · SO SÁNH MÔ HÌNH', 'BENCHMARKS · MODEL COMPARISON')} title={copy(language, 'So sánh & đề xuất', 'Comparison & recommendations')} subtitle={copy(language, 'Đối chiếu các model trong cùng bộ dữ liệu, điều kiện và quy trình đánh giá.', 'Compare models within the same dataset, condition, and evaluation protocol.')} action={selectedExperiment?.available && !isReference ? <a className="button button-outline" href={api.exportUrl(experimentId)}><ArrowDownToLine size={16} /> {copy(language, 'Xuất CSV', 'Export CSV')}</a> : undefined} />
    <div className="filter-bar panel"><div className="field"><label htmlFor="dataset-filter">{copy(language, 'Bộ dữ liệu & quy trình', 'Dataset & protocol')}</label><DatasetSelector experiments={experiments} value={experimentId} onChange={changeExperiment} /></div><div className="field"><label htmlFor="condition-filter">{copy(language, 'Điều kiện ảnh', 'Image condition')}</label><select id="condition-filter" value={activeCondition} onChange={(event) => { const next = event.target.value; setCondition(next); setStage(next === 'clean' ? 'clean' : stage === 'clean' ? 'raw' : stage); }}>{conditions.map((item) => <option key={item} value={item}>{rows.find((row) => row.condition === item)?.condition_label_bilingual.split(' · ')[language === 'vi' ? 0 : 1] ?? item}</option>)}</select></div><div className="field"><label htmlFor="stage-filter">{copy(language, 'Trạng thái ảnh', 'Image stage')}</label><select id="stage-filter" value={activeStage} onChange={(event) => setStage(event.target.value)}><option value="all">{copy(language, 'Tất cả trạng thái có dữ liệu', 'All available stages')}</option>{activeStages.map((item) => <option key={item} value={item}>{item === 'clean' ? copy(language, 'Ảnh gốc', 'Clean') : item === 'raw' ? copy(language, 'Ảnh nhiễu', 'Raw / noisy') : copy(language, 'Đã khôi phục', 'Restored')}</option>)}</select></div><div className="field"><label htmlFor="metric-filter">{copy(language, 'Chỉ số đo', 'Metric')}</label><select id="metric-filter" value={metric} onChange={(event) => setMetric(event.target.value as MetricKey)}>{METRIC_OPTIONS.map((item) => <option key={item.id} value={item.id}>{item[language]}</option>)}</select></div><button className="button button-outline" onClick={() => { const clean = conditions.includes('clean'); const nextExperiment = selectedExperiment; setCondition(clean ? 'clean' : conditions[0] ?? ''); setModel('all'); setStage(clean ? 'clean' : 'raw'); setMetric(nextExperiment?.map50_95_count ? 'mAP50_95' : nextExperiment?.map50_count ? 'mAP50' : 'mAP50_95'); }} disabled={loading}>{copy(language, 'Đặt lại bộ lọc', 'Reset filters')}</button></div>
    {selectedExperiment && <div className="provenance-strip"><Database size={15} /><div><strong>{selectedExperiment.title}</strong><span>{selectedExperiment.protocol}</span><small>Source: {selectedExperiment.source}</small></div><span className="provenance-count">{rows[0]?.images ? `${formatNumber(rows[0].images, 0)} ảnh` : 'n ảnh chưa xác nhận'}</span></div>}
    <section className="panel recommendation-panel"><div className="panel-heading"><div><div className="section-kicker">{copy(language, 'GỢI Ý DỰA TRÊN KẾT QUẢ ĐO', 'EVIDENCE-BASED MODEL PICK')}</div><h2>{copy(language, 'Gợi ý theo tiêu chí', 'Recommendation criterion')}</h2></div><label className="sr-only" htmlFor="recommendation-criterion">{copy(language, 'Tiêu chí gợi ý', 'Recommendation criterion')}</label><select id="recommendation-criterion" aria-label={copy(language, 'Tiêu chí gợi ý model', 'Recommendation criterion')} value={criterion} onChange={(event) => setCriterion(event.target.value)}><option value="map">{copy(language, 'mAP cao nhất', 'Highest mAP')}</option><option value="precision">{copy(language, 'Precision cao nhất', 'Highest precision')}</option><option value="recall">{copy(language, 'Recall cao nhất', 'Highest recall')}</option><option value="realtime">{copy(language, 'Nhanh nhất', 'Lowest latency')}</option><option value="robustness">{copy(language, 'Ít suy giảm nhất', 'Best robustness')}</option><option value="recovery">{copy(language, 'Khôi phục tốt nhất', 'Best recovery')}</option></select></div>
      {recommendation?.status === 'ok' && availableRows.length > 0 && model === 'all' ? <><div className="recommendation-winner"><div><small>{copy(language, 'Đứng đầu trong bộ lọc hiện tại', 'Top result for current filters')}</small><strong>{recommendation.winner?.model}</strong></div><b>{criterion === 'robustness' ? `${formatNumber(recommendation.winner?.value, 2)}% drop` : criterion === 'recovery' ? `${formatNumber((recommendation.winner?.value ?? 0) * 100, 2)} mAP pt` : criterion === 'realtime' ? `${formatNumber(recommendation.winner?.value, 2)} ms` : <>{recommendation.winner?.metric === 'mAP50' ? 'mAP@0.5 · ' : 'mAP@0.5:0.95 · '}{formatMetric(recommendation.winner?.value, 2)}</>}</b></div><p className="recommendation-note">{copy(language, 'Xếp hạng chỉ áp dụng cho dataset, evaluator, ngưỡng, điều kiện và protocol đang chọn.', 'Ranking is limited to the selected dataset, evaluator, thresholds, condition, and protocol.')} {copy(language, 'Nguồn:', 'Source:')} {recommendation.winner?.source_file} · {recommendation.winner?.images ? `${formatNumber(recommendation.winner.images, 0)} ${copy(language, 'ảnh', 'images')}` : copy(language, 'chưa rõ số ảnh', 'image count unavailable')}.</p></> : <div className="no-chart-data recommendation-empty"><Info size={20} /><strong>{availableRows.length === 0 ? copy(language, 'Không có kết quả khớp bộ lọc hiện tại', 'No results match the current filters') : model !== 'all' ? copy(language, 'Chọn tất cả model để xem gợi ý so sánh', 'Select all models to view a comparison recommendation') : selectedExperiment?.reference_only ? copy(language, 'Bộ dữ liệu này chỉ có số liệu tham khảo chưa xác minh.', 'This dataset contains unverified reference figures only.') : !selectedExperiment?.map50_95_count && selectedExperiment?.map50_count && (selectedExperiment?.model_count ?? 0) > 1 ? copy(language, 'Có mAP@0.5, nhưng chưa đủ kết quả chung điều kiện để xếp hạng.', 'mAP@0.5 is available, but there are not enough results under the same condition to rank.') : (selectedExperiment?.model_count ?? 0) < 2 ? copy(language, 'Lần đánh giá này chỉ có một model nên chưa thể so sánh để gợi ý.', 'This evaluation has only one model, so it cannot provide a comparison recommendation.') : recommendation?.reason ?? copy(language, 'Chưa đủ kết quả cùng protocol để xếp hạng.', 'Not enough results under the same protocol to rank models.')}</strong><span>{copy(language, 'Metric khả dụng và khả năng so sánh được ghi dưới bộ chọn.', 'Available metrics and comparison coverage are listed below the selector.')}</span></div>}
    </section>
    {error && <StateNotice icon={Info} title={copy(language, 'Không tải được dữ liệu benchmark', 'Could not load benchmark data')} kind="error">{error}. {copy(language, 'Kiểm tra backend rồi thử lại.', 'Check the backend and try again.')}</StateNotice>}
    {loading && !data && <StateNotice icon={Activity} title={copy(language, 'Đang đọc kết quả thực nghiệm', 'Loading experiment results')} kind="loading">{copy(language, 'Đang nạp CSV/metadata đã lưu; không chạy lại benchmark.', 'Loading saved CSV/metadata; the benchmark is not being rerun.')}</StateNotice>}
    {isReference && <StateNotice title={copy(language, 'Dữ liệu tham chiếu chưa xác minh', 'Unverified reference data')} kind="warning">{copy(language, 'Bảng này được cung cấp trong đặc tả. Điều kiện, dataset và evaluator chưa được xác nhận; số liệu không được dùng để xếp hạng.', 'This table came from the supplied specification. Its condition, dataset, and evaluator are unverified; the figures are excluded from rankings.')}</StateNotice>}
    {!error && data && <div className={`dataset-content ${loading ? 'is-updating' : ''}`} aria-busy={loading}>
      <div className="comparison-summary"><div className="summary-item"><span>{copy(language, 'Kết quả', 'Records')}</span><strong>{formatNumber(availableRows.length, 0)}</strong></div><div className="summary-item"><span>{copy(language, 'Dataset', 'Dataset')}</span><strong>{selectedExperiment?.dataset ?? '—'}</strong></div><div className="summary-item"><span>{copy(language, 'Mô hình', 'Models')}</span><strong>{new Set(availableRows.map((row) => row.model)).size}</strong></div><div className="summary-item"><span>{copy(language, 'Metric đang xem', 'Metric shown')}</span><strong>{metricLabel}</strong></div></div>
      <section className="panel comparison-chart-panel"><div className="panel-heading"><div><div className="section-kicker">{copy(language, 'KẾT QUẢ ĐO', 'MEASURED RESULTS')} · {activeCondition.toUpperCase()}</div><h2>{metricLabel} {copy(language, 'theo mô hình', 'by model')} <span>Model comparison</span></h2></div><select aria-label={copy(language, 'Lọc mô hình', 'Filter models')} value={model} onChange={(event) => setModel(event.target.value)} className="select-compact"><option value="all">{copy(language, 'Tất cả mô hình', 'All models')}</option>{models.map((item) => <option value={item} key={item}>{item}</option>)}</select></div>
        {chartData.length && chartData.some((row) => row.value != null) ? <div className="comparison-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData} margin={{ top: 18, right: 18, left: 3, bottom: 5 }}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8edf3" /><XAxis dataKey="model" tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} /><YAxis domain={metric === 'latency_ms' ? [0, 'auto'] : [0, 1]} tickFormatter={(value) => metric === 'latency_ms' ? `${value}` : `${Math.round(value * 100)}%`} tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} /><Tooltip formatter={(value) => metric === 'latency_ms' ? `${formatNumber(Number(value), 2)} ms` : formatMetric(Number(value), 2)} labelFormatter={(label) => String(label)} /><Bar dataKey="value" name={metricLabel} radius={[5, 5, 0, 0]}>{chartData.map((row, index) => <Cell key={`${row.model}-${index}`} fill={MODEL_COLORS[index % MODEL_COLORS.length]} />)}</Bar></BarChart></ResponsiveContainer></div> : <div className="no-chart-data"><BarChart3 size={22} /><strong>{copy(language, 'Chỉ số này chưa có trong nguồn đã chọn', 'This metric is not available in the selected source')}</strong><span>{copy(language, 'Chọn metric khác hoặc protocol có số liệu đo tương ứng.', 'Choose another metric or a protocol with the corresponding measurements.')}</span></div>}
      </section>
      <section className="panel table-panel"><div className="panel-heading"><div><div className="section-kicker">{copy(language, 'BẢN GHI NGUỒN', 'SOURCE RECORDS')} · {availableRows.length} {copy(language, 'DÒNG', 'ROWS')}</div><h2>{copy(language, 'Bảng kết quả', 'Results table')} <span>Metrics table</span></h2></div><label className="inline-filter">{copy(language, 'Mô hình', 'Model')}<select value={model} onChange={(event) => setModel(event.target.value)}><option value="all">{copy(language, 'Tất cả', 'All')}</option>{models.map((item) => <option key={item} value={item}>{item}</option>)}</select></label></div>
        <div className="table-scroll"><table className="data-table"><thead><tr><th>{copy(language, 'Mô hình', 'Model')}</th><th>{copy(language, 'Điều kiện / trạng thái', 'Condition / stage')}</th><th>mAP@0.5:0.95</th><th>mAP@0.5</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP / FP / FN</th><th>Latency</th><th>{copy(language, 'Ảnh', 'Images')}</th></tr></thead><tbody>{availableRows.map((row, index) => <tr key={`${row.model}-${row.condition}-${row.stage}-${index}`}><td><strong>{row.model}</strong><small>{row.input_size ? `img ${row.input_size}` : 'input size —'}</small></td><td>{row.condition_label_bilingual}<small>{row.stage}</small></td><td>{formatMetric(row.mAP50_95)}</td><td>{formatMetric(row.mAP50)}</td><td>{formatMetric(row.precision)}</td><td>{formatMetric(row.recall)}</td><td>{formatMetric(row.f1)}</td><td className="count-cell">{row.tp == null ? '—' : `${formatNumber(row.tp, 0)} / ${formatNumber(row.fp, 0)} / ${formatNumber(row.fn, 0)}`}</td><td>{row.latency_ms == null ? '—' : `${formatNumber(row.latency_ms, 2)} ms`}</td><td>{row.images == null ? '—' : formatNumber(row.images, 0)}</td></tr>)}</tbody></table></div>
        {!availableRows.length && <div className="no-chart-data"><Database size={22} /><strong>{copy(language, 'Không có kết quả khớp bộ lọc', 'No records match these filters')}</strong><span>{copy(language, 'Chọn điều kiện hoặc trạng thái ảnh khác.', 'Choose another condition or image stage.')}</span></div>}
        <div className="bottom-note"><Info size={15} /><span>{selectedExperiment?.protocol} {copy(language, 'Nguồn:', 'Source:')} {selectedExperiment?.source}. {copy(language, 'Precision, Recall và F1 có thể tính từ TP/FP/FN nếu CSV không lưu sẵn. F1 không được suy ra từ AP.', 'Precision, Recall, and F1 may be derived from TP/FP/FN when absent from the CSV. F1 is not derived from AP.')}</span></div>
      </section>
    </div>}
  </>;
}

function AnalyticsPage() {
  const language = useLanguage();
  const { experiments, experimentId, setExperimentId, data, loading, error } = useExperimentData('coco_normalized_320');
  const [model, setModel] = useState('YOLOv8n');
  const [family, setFamily] = useState('gaussian');
  const [stage, setStage] = useState('raw');
  const [metric, setMetric] = useState<'mAP50_95' | 'mAP50'>('mAP50_95');
  const rows = data?.records ?? [];
  const models = data?.filters.models ?? [];
  const families = data?.filters.families ?? [];
  const distortionFamilies = families.filter((item) => item !== 'clean');
  const activeFamily = distortionFamilies.includes(family) ? family : distortionFamilies[0] ?? 'clean';
  const familyStages = [...new Set(rows.filter((row) => row.family === activeFamily && row.model === model).map((row) => row.stage))];
  const activeStage = familyStages.includes(stage) ? stage : familyStages[0] ?? 'clean';
  const activeMetric = rows.some((row) => row[metric] != null) ? metric : rows.some((row) => row.mAP50_95 != null) ? 'mAP50_95' : 'mAP50';
  const cleanOnly = distortionFamilies.length === 0;
  const selectedExperiment = loading && data ? data.experiment : experiments.find((item) => item.id === experimentId) ?? data?.experiment;
  const curveRows = rows.filter((row) => row.model === model && (row.family === activeFamily || row.family === 'clean') && (row.family === 'clean' || row.stage === activeStage) && row[activeMetric] != null)
    .map((row) => ({ ...row, x: row.family === 'clean' ? 0 : row.severity ?? 0, value: row[activeMetric] as number, uncertainty: activeMetric === 'mAP50' ? row.mAP50_std ?? 0 : row.mAP50_95_std ?? 0 }))
    .sort((a, b) => a.x - b.x);
  const recoveryRows = rows.filter((restored) => restored.model === model && restored.family === activeFamily && restored.stage === 'restored').flatMap((restored) => {
    const raw = rows.find((candidate) => candidate.model === model && candidate.condition === restored.condition && candidate.family === activeFamily && candidate.stage === 'raw');
    const rawValue = raw?.[activeMetric]; const recoveredValue = restored[activeMetric];
    return raw && rawValue != null && recoveredValue != null ? [{ level: restored.severity ?? 0, gain: recoveredValue - rawValue }] : [];
  }).sort((a, b) => a.level - b.level);
  const errorRows = rows.filter((row) => row.model === model && (row.family === activeFamily || row.family === 'clean') && (row.stage === activeStage || row.family === 'clean') && row.tp != null && row.fp != null && row.fn != null);
  const isReference = data?.reference_only ?? false;

  useEffect(() => { if (models.length && !models.includes(model)) setModel(models[0]); }, [experimentId, models.join('|')]);
  function changeExperiment(value: string) {
    const next = experiments.find((item) => item.id === value);
    setExperimentId(value); setFamily('gaussian'); setStage('raw');
    setMetric(next?.map50_95_count ? 'mAP50_95' : next?.map50_count ? 'mAP50' : 'mAP50_95');
  }

  return <>
    <PageHeading eyebrow={copy(language, 'ROBUSTNESS · THÍ NGHIỆM', 'ROBUSTNESS · EXPERIMENTS')} title={copy(language, 'Phân tích nghiên cứu', 'Research analytics')} subtitle={copy(language, 'Theo dõi suy giảm, phục hồi và lỗi trên dữ liệu có nhãn.', 'Track degradation, recovery, and errors on labeled data.')} action={selectedExperiment?.available && !isReference ? <a className="button button-outline" href={api.exportUrl(experimentId)}><ArrowDownToLine size={16} /> {copy(language, 'Xuất CSV', 'Export CSV')}</a> : undefined} />
    <div className="filter-bar panel analytics-filters"><div className="field"><label htmlFor="analytics-dataset">{copy(language, 'Bộ dữ liệu & quy trình', 'Dataset & protocol')}</label><DatasetSelector id="analytics-dataset" experiments={experiments} value={experimentId} onChange={changeExperiment} /></div><div className="field"><label htmlFor="analytics-model">{copy(language, 'Mô hình', 'Model')}</label><select id="analytics-model" value={model} onChange={(event) => setModel(event.target.value)}>{models.map((item) => <option key={item}>{item}</option>)}</select></div><div className="field"><label htmlFor="analytics-family">{copy(language, 'Loại biến dạng', 'Distortion family')}</label><select id="analytics-family" value={activeFamily} disabled={cleanOnly} onChange={(event) => setFamily(event.target.value)}>{(cleanOnly ? ['clean'] : distortionFamilies).map((item) => <option key={item} value={item}>{translateFamily(item, language)}</option>)}</select></div><div className="field"><label htmlFor="analytics-stage">{copy(language, 'Trạng thái ảnh', 'Image stage')}</label><select id="analytics-stage" value={activeStage} disabled={cleanOnly || familyStages.length < 2} onChange={(event) => setStage(event.target.value)}>{familyStages.map((item) => <option key={item} value={item}>{item === 'clean' ? copy(language, 'Ảnh gốc', 'Clean') : item === 'raw' ? copy(language, 'Ảnh nhiễu', 'Raw / noisy') : copy(language, 'Đã khôi phục', 'Restored')}</option>)}</select></div><div className="field"><label htmlFor="analytics-metric">{copy(language, 'Chỉ số', 'Metric')}</label><select id="analytics-metric" value={activeMetric} onChange={(event) => setMetric(event.target.value as 'mAP50_95' | 'mAP50')}><option value="mAP50_95" disabled={!rows.some((row) => row.mAP50_95 != null)}>mAP@0.5:0.95 · {copy(language, 'độ chính xác trung bình', 'mean average precision')}</option><option value="mAP50" disabled={!rows.some((row) => row.mAP50 != null)}>mAP@0.5 · {copy(language, 'độ chính xác IoU 0,5', 'precision at IoU 0.5')}</option></select></div></div>
    {selectedExperiment && <div className="provenance-strip"><Database size={15} /><div><strong>{selectedExperiment.title}</strong><span>{selectedExperiment.protocol}</span><small>{copy(language, 'Nguồn:', 'Source:')} {selectedExperiment.source}</small></div><span className="provenance-count">{rows[0]?.subset_count ? `${rows[0].subset_count} ${copy(language, 'tập con', 'subsets')}` : `${formatNumber(rows[0]?.images, 0)} ${copy(language, 'ảnh', 'images')}`}</span></div>}
    {error && <StateNotice icon={Info} title={copy(language, 'Không tải được dữ liệu', 'Could not load data')} kind="error">{error}</StateNotice>}
    {loading && !data && <StateNotice icon={Activity} title={copy(language, 'Đang đọc kết quả thực nghiệm', 'Loading experiment results')} kind="loading">{copy(language, 'Đang nạp số liệu và provenance từ CSV.', 'Loading metrics and provenance from CSV.')}</StateNotice>}
    {isReference && <StateNotice title={copy(language, 'Chưa có protocol đủ tin cậy', 'No verified protocol available')} kind="warning">{copy(language, 'Tài liệu tham chiếu chưa xác nhận dataset, điều kiện, số ảnh và evaluator; không thể dùng để phân tích suy giảm, khôi phục hoặc gợi ý model.', 'The reference has no confirmed dataset, condition, image count, or evaluator; it cannot support degradation, recovery, or model recommendations.')}</StateNotice>}
    {data && !error && !isReference && <div className={`dataset-content ${loading ? 'is-updating' : ''}`} aria-busy={loading}>
      <div className="analytics-intro"><div className="analytics-icon"><Activity size={19} /></div><div><strong>{copy(language, 'Đọc kết quả theo protocol đã chọn', 'Read results under the selected protocol')}</strong><p>{copy(language, 'Đường cong chỉ dùng những mức đã đo. Độ lệch chuẩn chỉ xuất hiện cho aggregate có nhiều subset; không dựng khoảng tin cậy từ một lần chạy.', 'Curves use measured severity levels only. Standard deviation appears only for multi-subset aggregates; no confidence interval is inferred from a single run.')}</p></div></div>
      {cleanOnly && <StateNotice icon={Info} title={copy(language, 'Dataset này chỉ có ảnh sạch', 'This dataset contains clean images only')} kind="neutral">{copy(language, 'Không có severity để vẽ đường suy giảm hoặc cặp raw/restored để đo khôi phục. Chỉ các metric và count có trong bảng kết quả mới được hiển thị.', 'There are no severity levels for a degradation curve or raw/restored pairs for recovery analysis. Only metrics and counts present in the source records are shown.')}</StateNotice>}
      <div className="analytics-grid">
        <section className="panel chart-panel wide"><div className="panel-heading"><div><div className="section-kicker">{copy(language, 'ĐƯỜNG SUY GIẢM', 'DEGRADATION CURVE')} · {activeStage.toUpperCase()}</div><h2>{activeMetric === 'mAP50' ? 'mAP@0.5' : 'mAP@0.5:0.95'} {copy(language, 'theo cường độ', 'vs severity')} <span>mAP vs severity</span></h2></div><span className="chart-legend"><i /> Clean baseline · {model}</span></div>
          {cleanOnly ? <ChartEmpty label={copy(language, 'Dataset không có ảnh nhiễu để vẽ đường suy giảm.', 'This dataset has no distorted images for a degradation curve.')} /> : curveRows.length > 1 ? <div className="research-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={curveRows} margin={{ top: 15, right: 20, left: 5, bottom: 8 }}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8edf3" /><XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(value) => Number(value).toFixed(2)} label={{ value: 'Severity / σ', position: 'insideBottom', offset: -4, fontSize: 11 }} tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} /><YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} /><Tooltip labelFormatter={(value) => `Severity ${Number(value).toFixed(2)}`} formatter={(value) => formatMetric(Number(value), 2)} /><Line type="monotone" dataKey="value" name={activeMetric === 'mAP50' ? 'mAP@0.5' : 'mAP@0.5:0.95'} stroke="#7c3aed" strokeWidth={2.5} dot={{ r: 4, fill: '#7c3aed' }} activeDot={{ r: 6 }}><ErrorBar dataKey="uncertainty" width={4} strokeWidth={1} stroke="#a78bfa" /></Line></LineChart></ResponsiveContainer></div> : <ChartEmpty label={copy(language, 'Chưa có đủ điểm severity trong experiment này.', 'This experiment has too few measured severity points.')} />}
        </section>
        <section className="panel chart-panel"><div className="panel-heading"><div><div className="section-kicker">{copy(language, 'HIỆU QUẢ KHÔI PHỤC', 'RECOVERY EFFECT')}</div><h2>{copy(language, 'Mức tăng sau khôi phục', 'Recovery gain')} <span>Restored − raw</span></h2></div><button className="chart-more" title={copy(language, 'Gain hiển thị theo điểm mAP tuyệt đối', 'Gain is shown in absolute mAP points')} aria-label={copy(language, 'Giải thích mức tăng khôi phục', 'Explain recovery gain')}><CircleHelp size={16} /></button></div>
          {recoveryRows.length ? <div className="mini-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={recoveryRows} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8edf3" /><XAxis dataKey="level" tickFormatter={(value) => Number(value).toFixed(2)} tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} /><YAxis tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)} pt`} tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} /><Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(2)} mAP points`} /><Bar dataKey="gain" name="Recovery gain" fill="#06b6d4" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></div> : <ChartEmpty label={copy(language, 'Chưa có cặp ảnh nhiễu / khôi phục trong protocol này.', 'No noisy/restored image pairs are available in this protocol.')} compact />}
        </section>
        <section className="panel chart-panel"><div className="panel-heading"><div><div className="section-kicker">TP / FP / FN · {copy(language, 'DỮ LIỆU CÓ NHÃN', 'LABELED DATA')}</div><h2>{copy(language, 'Lỗi theo điều kiện', 'Errors by condition')} <span>Matching counts</span></h2></div></div>
          {errorRows.length ? <div className="table-scroll"><table className="data-table compact-table"><thead><tr><th>{copy(language, 'Điều kiện', 'Condition')}</th><th>TP</th><th>FP</th><th>FN</th></tr></thead><tbody>{errorRows.map((row, index) => <tr key={`${row.condition}-${index}`}><td>{row.condition_label_bilingual}<small>{row.stage}</small></td><td>{formatNumber(row.tp, 0)}</td><td className="error-fp">{formatNumber(row.fp, 0)}</td><td className="error-fn">{formatNumber(row.fn, 0)}</td></tr>)}</tbody></table></div> : <ChartEmpty label={copy(language, 'Chưa có count matching hợp lệ cho bộ lọc này.', 'No valid matching counts for these filters.')} compact />}
        </section>
      </div>
      {rows.some((row) => row.subset_count && row.mAP50_95_std != null) && <section className="panel table-panel subset-panel"><div className="panel-heading"><div><div className="section-kicker">{copy(language, 'ĐỘ PHÂN TÁN GIỮA CÁC TẬP CON', 'SUBSET VARIANCE')}</div><h2>{copy(language, 'Trung bình ± sample SD', 'Mean ± sample SD')} <span>n = 3 disjoint subsets</span></h2></div></div><div className="table-scroll"><table className="data-table"><thead><tr><th>{copy(language, 'Điều kiện', 'Condition')}</th><th>mAP@0.5:0.95</th><th>mAP@0.5</th><th>{copy(language, 'Thay đổi tương đối', 'Relative change')}</th><th>{copy(language, 'Tập con', 'Subsets')}</th><th>{copy(language, 'Ảnh / tập con', 'Images / subset')}</th></tr></thead><tbody>{rows.filter((row) => row.subset_count).map((row, index) => <tr key={`${row.condition}-${index}`}><td>{row.condition_label_bilingual}<small>{row.stage}</small></td><td>{formatMetric(row.mAP50_95)} ± {formatMetric(row.mAP50_95_std)}</td><td>{formatMetric(row.mAP50)} ± {formatMetric(row.mAP50_std)}</td><td>{formatNumber(row.relative_change_vs_clean_pct, 2)}% ± {formatNumber(row.relative_change_vs_clean_pct_std, 2)} pp</td><td>{row.subset_count}</td><td>{formatNumber(row.images, 0)}</td></tr>)}</tbody></table></div><div className="bottom-note"><Info size={15} /><span>{copy(language, 'SD là sample standard deviation trong file aggregate. Mỗi tập con gồm 500 ảnh; cần giữ nguyên manifest ID để tái lập.', 'SD is the sample standard deviation in the aggregate file. Each subset has 500 images; preserve the manifest IDs for reproducibility.')}</span></div></section>}
      <StateNotice title={copy(language, 'Cách đọc chỉ số', 'Metric notes')}>Degradation = mAP_clean − mAP_noisy ({copy(language, 'điểm mAP', 'mAP points')}). Relative degradation = 100 × (clean − noisy) / clean (%). Recovery gain = mAP_restored − mAP_noisy. {copy(language, 'Không có ground truth thì không tính mAP hoặc confusion counts.', 'Without ground truth, mAP and confusion counts cannot be calculated.')}</StateNotice>
    </div>}
  </>;
}

function ChartEmpty({ label, compact = false }: { label: string; compact?: boolean }) {
  const language = useLanguage();
  return <div className={`no-chart-data ${compact ? 'compact-empty' : ''}`}><BarChart3 size={22} /><strong>{label}</strong><span>{copy(language, 'Không thay bằng dữ liệu minh họa.', 'No illustrative data is substituted.')}</span></div>;
}

function MethodologyPage() {
  const language = useLanguage();
  const [tourVisible, setTourVisible] = useState(true);
  return <>
    <PageHeading eyebrow={copy(language, 'TÍNH TÁI LẬP · NGUỒN DỮ LIỆU', 'REPRODUCIBILITY · DATA SOURCES')} title={copy(language, 'Phương pháp & thư viện', 'Methodology & gallery')} subtitle={copy(language, 'Cách tạo biến dạng, đo hiệu năng và đọc giới hạn thí nghiệm.', 'How distortions are generated, performance is measured, and study limitations are interpreted.')} />
    {tourVisible && <div className="tour-banner"><div className="tour-icon"><Sparkles size={18} /></div><div><strong>{copy(language, 'Bắt đầu với research lab', 'Start with the research lab')}</strong><p>{copy(language, 'Chọn dataset có nhãn → so sánh ảnh gốc và ảnh biến dạng → xem mức cải thiện sau khôi phục → kiểm tra nguồn dữ liệu.', 'Choose a labeled dataset → compare clean and distorted images → review restoration gains → check data provenance.')}</p></div><button aria-label={copy(language, 'Ẩn hướng dẫn nhanh', 'Hide quick guide')} className="icon-button" onClick={() => setTourVisible(false)}><X size={17} /></button></div>}
    <section className="panel methodology-panel"><div className="panel-heading"><div><div className="section-kicker">EXPERIMENT PIPELINE</div><h2>{copy(language, 'Quy trình thực nghiệm', 'Experiment pipeline')} <span>{copy(language, 'Từ dữ liệu đến bằng chứng', 'From data to evidence')}</span></h2></div></div><div className="pipeline-steps">
      {[
        ['01', 'Chuẩn bị dữ liệu', 'COCO val2017 có nhãn; các subset được khóa bằng manifest và seed.'],
        ['02', 'Đánh giá baseline', 'YOLOv8n, YOLOv8s và SSD-MobileNet pretrained; không fine-tune trong ứng dụng.'],
        ['03', 'Tạo biến dạng', 'Gaussian, Salt-and-Pepper, fog và low-light tổng hợp; seed và severity phải được ghi nhận.'],
        ['04', 'Khôi phục ảnh', 'NLM là nhánh khử Gaussian; median, CLAHE và bilateral được đối chiếu theo protocol.'],
        ['05', 'Đo & phân tích', 'Cùng image IDs, metric implementation và thresholds; paired comparison khi dữ liệu cho phép.'],
      ].map(([number, title, body], index) => <div className="pipeline-step" key={number}><div className="step-number">{number}</div><div className="step-body"><strong>{language === 'vi' ? title : ({ 'Chuẩn bị dữ liệu': 'Prepare data', 'Đánh giá baseline': 'Evaluate baseline', 'Tạo biến dạng': 'Generate distortions', 'Khôi phục ảnh': 'Restore images', 'Đo & phân tích': 'Measure & analyze' } as Record<string, string>)[title]}</strong><span>{language === 'vi' ? body : ({ 'COCO val2017 có nhãn; các subset được khóa bằng manifest và seed.': 'Labeled COCO val2017; subsets are fixed by manifest and seed.', 'YOLOv8n, YOLOv8s và SSD-MobileNet pretrained; không fine-tune trong ứng dụng.': 'Pretrained YOLOv8n, YOLOv8s, and SSD-MobileNet; no fine-tuning in the app.', 'Gaussian, Salt-and-Pepper, fog và low-light tổng hợp; seed và severity phải được ghi nhận.': 'Synthetic Gaussian, salt-and-pepper, fog, and low-light; record seed and severity.', 'NLM là nhánh khử Gaussian; median, CLAHE và bilateral được đối chiếu theo protocol.': 'NLM denoises Gaussian noise; median, CLAHE, and bilateral are compared by protocol.', 'Cùng image IDs, metric implementation và thresholds; paired comparison khi dữ liệu cho phép.': 'Use the same image IDs, metric implementation, and thresholds; pair comparisons when possible.' } as Record<string, string>)[body]}</span></div>{index < 4 && <div className="step-connector" />}</div>)}
    </div></section>
    <div className="two-column">
      <section id="provenance" className="panel methodology-panel"><div className="section-kicker">{copy(language, 'NGUỒN DỮ LIỆU', 'DATA PROVENANCE')}</div><h2>{copy(language, 'Nguồn dữ liệu', 'Data sources')} <span>What is measured</span></h2><div className="source-list"><div><Database size={17} /><span><strong>COCO val2017</strong><small>{copy(language, '5.000 ảnh có annotation · benchmark chính', '5,000 annotated images · primary benchmark')}</small></span><Check size={15} className="source-check" /></div><div><Database size={17} /><span><strong>3 × COCO subsets</strong><small>{copy(language, '500 ảnh mỗi subset · dùng aggregate đã ghi nhận', '500 images per subset · recorded aggregates only')}</small></span><Check size={15} className="source-check" /></div><div><Database size={17} /><span><strong>Open Images V7</strong><small>{copy(language, '100 + 300 ảnh · dataset ngoài, class mapping riêng', '100 + 300 images · external dataset with separate class mapping')}</small></span><Check size={15} className="source-check" /></div></div><StateNotice icon={Info} title={copy(language, 'Không so sánh chéo protocol', 'Do not compare across protocols')}>{copy(language, 'COCO-80 mAP và Open Images subset mAP dùng dataset, class coverage và evaluator khác nhau; không xếp hạng chung.', 'COCO-80 mAP and Open Images subset mAP use different datasets, class coverage, and evaluators; do not rank them together.')}</StateNotice></section>
      <section className="panel methodology-panel"><div className="section-kicker">{copy(language, 'THUẬT NGỮ METRIC', 'METRIC GLOSSARY')}</div><h2 id="glossary">{copy(language, 'Thuật ngữ', 'Glossary')} <span>Glossary</span></h2><div className="glossary-list">
        {[
           ['mAP', 'mean Average Precision', copy(language, 'Trung bình AP theo lớp; phụ thuộc IoU và protocol.', 'Mean AP across classes; depends on IoU and protocol.')],
           ['IoU', 'Intersection over Union', copy(language, 'Tỷ lệ diện tích giao trên diện tích hợp giữa hai hộp.', 'Intersection area divided by union area for two boxes.')],
           ['NMS', 'Non-Maximum Suppression', copy(language, 'Loại hộp trùng dựa trên confidence và IoU.', 'Suppresses overlapping boxes using confidence and IoU.')],
           ['NLM', 'Non-Local Means', copy(language, 'Khử nhiễu bằng cách lấy trung bình các vùng ảnh tương tự.', 'Denoises by averaging similar image regions.')],
           ['TP / FP / FN', 'True / False Positive · False Negative', copy(language, 'Phát hiện đúng, phát hiện nhầm và bỏ sót theo matching rule.', 'Correct detections, false detections, and missed objects under the matching rule.')],
           ['Confidence', 'Detection confidence', copy(language, 'Điểm tin cậy của detection; không phải xác suất đã hiệu chuẩn.', 'Detection score; not a calibrated probability.')],
        ].map(([term, english, definition]) => <div className="glossary-row" key={term}><div><strong>{term}</strong><small>{english}</small></div><p>{definition}</p></div>)}
      </div></section>
    </div>
    <GallerySection />
    <section className="panel limitations-panel"><div className="limitations-icon"><Info size={18} /></div><div><div className="section-kicker">{copy(language, 'GIỚI HẠN', 'LIMITATIONS')}</div><h2>{copy(language, 'Giới hạn cần nhớ', 'Limitations to keep in mind')}</h2><p>{copy(language, 'Biến dạng mô phỏng không thay thế thời tiết hoặc cảm biến ngoài đời thật. Kết quả trên subset không tương đương đánh giá toàn bộ COCO. Ảnh upload/webcam không có ground truth chỉ cho phép xem detections và latency, không tính mAP hoặc confusion counts.', 'Synthetic distortions do not reproduce real weather or sensor behavior. Subset results are not equivalent to evaluation on all of COCO. Uploads/webcam images without ground truth show detections and latency only; mAP and confusion counts cannot be calculated.')}</p></div></section>
  </>;
}

function translateClass(name: string, language: Language) {
  const vi: Record<string, string> = { Person: 'Người', Chair: 'Ghế', Couch: 'Sofa', Bed: 'Giường', 'Remote control': 'Điều khiển từ xa', 'Mobile phone': 'Điện thoại', Clock: 'Đồng hồ', 'Teddy bear': 'Gấu bông' };
  return language === 'vi' ? vi[name] ?? name : name;
}

function GallerySection() {
  const language = useLanguage();
  const [page, setPage] = useState(1);
  const [className, setClassName] = useState('');
  const [showGroundTruth, setShowGroundTruth] = useState(true);
  const [showPredictions, setShowPredictions] = useState(true);
  const [data, setData] = useState<GalleryResponse | null>(null);
  const [failedImages, setFailedImages] = useState<Set<string>>(() => new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    setLoading(true);
    setError('');
    setFailedImages(new Set());
    api.gallery(page, className).then((value) => { if (active) setData(value); }).catch((cause: Error) => { if (active) setError(cause.message); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [page, className]);
  return <section id="gallery" className="panel gallery-section"><div className="panel-heading"><div><div className="section-kicker">{copy(language, 'ẢNH MINH HỌA CÓ NHÃN', 'LABELED EXAMPLES')}</div><h2>{copy(language, 'Thư viện ảnh đối chiếu', 'Labeled image gallery')}</h2></div><div className="gallery-controls"><label className="sr-only" htmlFor="gallery-class">{copy(language, 'Lọc lớp', 'Filter class')}</label><select id="gallery-class" value={className} onChange={(event) => { setPage(1); setClassName(event.target.value); }}><option value="">{copy(language, 'Tất cả lớp', 'All classes')}</option>{(data?.classes ?? []).map((item) => <option key={item} value={item}>{translateClass(item, language)}</option>)}</select><span>{data?.total ?? 0} {copy(language, 'ảnh', 'images')}</span></div></div>
    <div className="gallery-legend"><label className="gallery-toggle"><input type="checkbox" checked={showGroundTruth} onChange={(event) => setShowGroundTruth(event.target.checked)} /><i className="legend-swatch legend-gt" /><span>{copy(language, 'Nhãn chuẩn', 'Ground truth')}</span></label><label className="gallery-toggle"><input type="checkbox" checked={showPredictions} onChange={(event) => setShowPredictions(event.target.checked)} /><i className="legend-swatch legend-pred" /><span>{copy(language, 'Dự đoán YOLO', 'YOLO predictions')}</span></label><span className="gallery-confidence">{copy(language, 'Conf ≥ 0,25 · ngưỡng hiển thị', 'Conf ≥ 0.25 · display filter')}</span></div>
    <div className="bottom-note"><Info size={15} /><span>{copy(language, 'Viền xanh liền là nhãn chuẩn; viền cam đứt là dự đoán YOLO. Ngưỡng confidence chỉ lọc hình vẽ, không tạo metric mới.', 'Green solid boxes show ground truth; orange dashed boxes show YOLO predictions. Confidence only filters displayed boxes, not benchmark metrics.')}</span></div>
    {loading && <StateNotice icon={Activity} title={copy(language, 'Đang tải ảnh có nhãn', 'Loading labeled images')} kind="loading">{copy(language, 'Đang nạp manifest, ground truth và prediction đã lưu.', 'Loading the manifest, ground truth, and saved predictions.')}</StateNotice>}
    {error && <StateNotice title={copy(language, 'Không tải được thư viện ảnh', 'Could not load the image gallery')} kind="error">{error}</StateNotice>}
    {!loading && !error && data && data.items.length === 0 && <StateNotice icon={Info} title={copy(language, 'Không có ảnh khớp bộ lọc', 'No images match this filter')} kind="neutral">{copy(language, 'Thử chọn lớp khác hoặc tải lại trang.', 'Choose another class or reload the page.')}</StateNotice>}
    <div className="gallery-grid">{!loading && !error && data?.items.map((item) => <article className="gallery-card" key={item.image_id}><div className="gallery-image" style={{ aspectRatio: item.image_size ? `${item.image_size.width}/${item.image_size.height}` : '4/3' }}>{failedImages.has(item.image_id) ? <div className="gallery-image-fallback">{copy(language, 'Không tải được ảnh · mở nguồn ảnh để xem', 'Image unavailable · open the source image')}</div> : <><img src={item.image_url} alt={copy(language, `Ảnh Open Images ${item.image_id}`, `Open Images ${item.image_id}`)} loading="lazy" onError={() => setFailedImages((current) => new Set(current).add(item.image_id))} onLoad={(event) => { const image = event.currentTarget; if (image.naturalWidth && image.naturalHeight) image.parentElement!.style.aspectRatio = `${image.naturalWidth}/${image.naturalHeight}`; }} />{showGroundTruth && item.ground_truth.map((box, index) => <div key={`gt-${index}`} className="gallery-box box-gt" style={{ left: `${box.x1 * 100}%`, top: `${box.y1 * 100}%`, width: `${(box.x2 - box.x1) * 100}%`, height: `${(box.y2 - box.y1) * 100}%` }}><span>GT · {translateClass(box.class_name, language)}{box.group_of ? ' · group' : ''}</span></div>)}{showPredictions && item.predictions.map((box, index) => <div key={`pred-${index}`} className="gallery-box box-pred" style={{ left: `${box.x1 * 100}%`, top: `${box.y1 * 100}%`, width: `${(box.x2 - box.x1) * 100}%`, height: `${(box.y2 - box.y1) * 100}%` }}><span>YOLO · {translateClass(box.class_name, language)} {box.confidence.toFixed(2)}</span></div>)}</>}</div><div className="gallery-card-body"><strong>{item.class_names.map((name) => translateClass(name, language)).join(' · ') || copy(language, 'Ảnh chưa gán nhãn', 'Unlabeled image')}</strong><small>{item.model} · {copy(language, `Nhãn chuẩn ${item.ground_truth.length} · dự đoán ${item.predictions.length}`, `Ground truth ${item.ground_truth.length} · predictions ${item.predictions.length}`)}</small><small>{copy(language, 'Tác giả', 'Author')}: {item.author || copy(language, 'chưa rõ', 'unknown')} · <a href={item.license} target="_blank" rel="noreferrer">{copy(language, 'Giấy phép', 'License')}</a></small><a className="gallery-source" href={item.source} target="_blank" rel="noreferrer">{copy(language, 'Nguồn ảnh', 'Image source')}</a></div></article>)}</div>
    {data && <div className="gallery-pagination"><button className="button button-outline" disabled={page <= 1} onClick={() => setPage(page - 1)}>← {copy(language, 'Trước', 'Previous')}</button><span>{copy(language, 'Trang', 'Page')} {page} / {Math.max(1, Math.ceil(data.total / data.page_size))}</span><button className="button button-outline" disabled={page * data.page_size >= data.total} onClick={() => setPage(page + 1)}>{copy(language, 'Tiếp', 'Next')} →</button></div>}
  </section>;
}

export default App;
