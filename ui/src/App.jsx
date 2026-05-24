import { useEffect, useRef, useState } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';

const COLORS = {
  bg: '#0f1117',
  surface: '#1a1d27',
  surfaceAlt: '#222633',
  border: '#2a2f3e',
  borderStrong: '#3a4055',
  accent: '#3b82f6',
  accentSoft: 'rgba(59, 130, 246, 0.12)',
  text: '#e4e7ec',
  textMuted: '#8a92a3',
  textDim: '#5f6779',
  success: '#22c55e',
  error: '#ef4444',
};

const EVENT_TYPES = [
  { value: 'basketball', label: 'Basketball' },
  { value: 'football',   label: 'Football'   },
  { value: 'concert',    label: 'Concert'    },
  { value: 'hockey',     label: 'Hockey'     },
  { value: 'tennis',     label: 'Tennis'     },
];

function humanize(slug) {
  if (!slug) return '';
  return slug.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function buildPromptPreview({ stadium, section, row, seat, eventType }) {
  if (!stadium) return '';
  return `a sov photo of ${stadium} stadium, section ${section || '_'} row ${row || '_'} seat ${seat || '_'}, ${eventType} event, photorealistic wide angle stadium view`;
}

export default function App() {
  const [status, setStatus]       = useState('checking');
  const [stadiums, setStadiums]   = useState([]);
  const [catalog, setCatalog]     = useState([]);
  const [stadium, setStadium]     = useState('');
  const [section, setSection]     = useState('122');
  const [row, setRow]             = useState('G');
  const [seat, setSeat]           = useState('14');
  const [eventType, setEventType] = useState('basketball');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [steps, setSteps]         = useState(20);
  const [guidance, setGuidance]   = useState(3.5);
  const [seed, setSeed]           = useState('');
  const [genState, setGenState]   = useState('idle');
  const [progress, setProgress]   = useState(0);
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);
  const [showPromptDetail, setShowPromptDetail] = useState(false);
  const progressTimer = useRef(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [healthRes, stadRes] = await Promise.all([
          fetch(`${API_BASE}/health`),
          fetch(`${API_BASE}/stadiums`),
        ]);
        if (!healthRes.ok) throw new Error('health not ok');
        const health = await healthRes.json();
        const stad = stadRes.ok ? await stadRes.json() : { stadiums: [] };
        if (cancelled) return;
        setStatus('ready');
        setStadiums(health.loaded_stadiums || []);
        const catalogList = Array.isArray(stad.stadiums) ? stad.stadiums : [];
        setCatalog(catalogList);
        const firstTrained = catalogList.find((s) => s.trained);
        const first = firstTrained || catalogList[0];
        if (first) setStadium((cur) => cur || first.slug);
      } catch {
        if (!cancelled) setStatus('offline');
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const promptPreview = buildPromptPreview({ stadium, section, row, seat, eventType });
  const selectedEntry = catalog.find((c) => c.slug === stadium);
  const selectedTrained = selectedEntry ? selectedEntry.trained : stadiums.includes(stadium);
  const canGenerate =
    status === 'ready' &&
    stadium && section && row && seat &&
    selectedTrained &&
    genState !== 'loading';

  function startProgress(estimatedSeconds) {
    setProgress(0);
    const start = Date.now();
    const total = estimatedSeconds * 1000;
    progressTimer.current = setInterval(() => {
      const elapsed = Date.now() - start;
      const fraction = Math.min(elapsed / total, 1);
      setProgress(Math.min(90, fraction * 90));
    }, 120);
  }
  function stopProgress() {
    if (progressTimer.current) {
      clearInterval(progressTimer.current);
      progressTimer.current = null;
    }
  }
  useEffect(() => stopProgress, []);

  async function handleGenerate() {
    if (!canGenerate) return;
    setGenState('loading');
    setError(null);
    setResult(null);
    setShowPromptDetail(false);

    const estimate = Math.max(8, steps * 1.5);
    startProgress(estimate);

    const body = {
      stadium,
      section: String(section),
      row: String(row),
      seat: String(seat),
      event_type: eventType,
      steps: Number(steps),
      guidance_scale: Number(guidance),
    };
    if (seed) body.seed = parseInt(seed, 10);

    try {
      const res = await fetch(`${API_BASE}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      stopProgress();
      setProgress(100);
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResult({
        imageBase64: data.image_base64,
        prompt: data.prompt_used,
        time: data.generation_time_seconds,
        stadium: data.stadium,
        section: data.section,
        row, seat,
      });
      setGenState('done');
    } catch (e) {
      stopProgress();
      setProgress(0);
      setError(e.message || 'Generation failed');
      setGenState('error');
    }
  }

  function handleDownload() {
    if (!result?.imageBase64) return;
    const a = document.createElement('a');
    a.href = `data:image/jpeg;base64,${result.imageBase64}`;
    a.download = `${result.stadium}_section${result.section}_row${result.row}_seat${result.seat}.jpg`;
    a.click();
  }

  return (
    <div style={S.app}>
      <StatsBar status={status} stadiumsCount={stadiums.length} />
      <div style={S.layout}>
        <ControlsPanel
          stadiums={stadiums}
          catalog={catalog}
          selectedTrained={selectedTrained}
          stadium={stadium}        setStadium={setStadium}
          section={section}        setSection={setSection}
          row={row}                setRow={setRow}
          seat={seat}              setSeat={setSeat}
          eventType={eventType}    setEventType={setEventType}
          showAdvanced={showAdvanced} setShowAdvanced={setShowAdvanced}
          steps={steps}            setSteps={setSteps}
          guidance={guidance}      setGuidance={setGuidance}
          seed={seed}              setSeed={setSeed}
          promptPreview={promptPreview}
          canGenerate={canGenerate}
          onGenerate={handleGenerate}
          status={status}
        />
        <OutputPanel
          genState={genState}
          progress={progress}
          result={result}
          error={error}
          showPromptDetail={showPromptDetail}
          setShowPromptDetail={setShowPromptDetail}
          onDownload={handleDownload}
        />
      </div>
    </div>
  );
}

function StatsBar({ status, stadiumsCount }) {
  const dotColor = status === 'ready' ? COLORS.success : status === 'offline' ? COLORS.error : COLORS.textDim;
  const dotLabel = status === 'ready' ? 'Ready' : status === 'offline' ? 'Offline' : 'Checking…';
  return (
    <div style={S.statsBar}>
      <div style={S.statsLeft}>
        <span style={S.brandMark}>◆</span>
        <span style={S.brandText}>attend</span>
      </div>
      <div style={S.statsCenter}>
        <Stat label="Stadiums" value={stadiumsCount} />
        <span style={S.statSep}>·</span>
        <Stat label="Model" value="FLUX.1-dev + LoRA" />
      </div>
      <div style={S.statsRight}>
        <span
          className={status === 'ready' ? 'pulse-dot' : ''}
          style={{ ...S.statusDot, background: dotColor, boxShadow: status === 'ready' ? `0 0 10px ${dotColor}` : 'none' }}
        />
        <span style={{ color: status === 'ready' ? COLORS.success : status === 'offline' ? COLORS.error : COLORS.textMuted, fontWeight: 500 }}>
          {dotLabel}
        </span>
      </div>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <span style={S.stat}>
      <span style={S.statLabel}>{label}:</span>
      <span style={S.statValue}>{value}</span>
    </span>
  );
}

function ControlsPanel({
  stadiums, catalog, selectedTrained,
  stadium, setStadium,
  section, setSection, row, setRow, seat, setSeat,
  eventType, setEventType,
  showAdvanced, setShowAdvanced,
  steps, setSteps, guidance, setGuidance, seed, setSeed,
  promptPreview, canGenerate, onGenerate, status,
}) {
  const trained = catalog.filter((c) => c.trained);
  const untrained = catalog.filter((c) => !c.trained);
  const fallbackOptions = catalog.length === 0 ? stadiums.map((s) => ({ slug: s, name: humanize(s), trained: true })) : [];

  const renderOption = (entry) => {
    const cityBit = entry.city ? ` — ${entry.city}` : '';
    return (
      <option key={entry.slug} value={entry.slug} disabled={!entry.trained && stadium !== entry.slug}>
        {(entry.name || humanize(entry.slug)) + cityBit + (entry.trained ? '' : '  (not trained)')}
      </option>
    );
  };

  return (
    <div style={S.controlsPanel}>
      <div>
        <h1 style={S.title}>attend — Venue Intelligence Platform</h1>
        <p style={S.subtitle}>LoRA + DreamBooth Seat View Generator</p>
      </div>

      <Field label="Stadium">
        <select
          className="field-select"
          style={S.input}
          value={stadium}
          onChange={(e) => setStadium(e.target.value)}
          disabled={catalog.length === 0 && fallbackOptions.length === 0}
        >
          {catalog.length === 0 && fallbackOptions.length === 0 ? (
            <option value="">{status === 'offline' ? 'API offline' : 'No stadiums available'}</option>
          ) : catalog.length === 0 ? (
            fallbackOptions.map(renderOption)
          ) : (
            <>
              {trained.length > 0 && (
                <optgroup label={`Trained (${trained.length})`}>
                  {trained.map(renderOption)}
                </optgroup>
              )}
              {untrained.length > 0 && (
                <optgroup label={`Available — not yet trained (${untrained.length})`}>
                  {untrained.map(renderOption)}
                </optgroup>
              )}
            </>
          )}
        </select>
        {stadium && !selectedTrained && status === 'ready' && (
          <span style={S.warnText}>
            No LoRA trained for this stadium yet — run scripts/train_lora.py to enable generation.
          </span>
        )}
      </Field>

      <div style={S.row3}>
        <Field label="Section">
          <input
            className="field-input"
            style={S.input}
            type="number"
            value={section}
            onChange={(e) => setSection(e.target.value)}
            placeholder="122"
          />
        </Field>
        <Field label="Row">
          <input
            className="field-input"
            style={S.input}
            type="text"
            maxLength={3}
            value={row}
            onChange={(e) => setRow(e.target.value.toUpperCase())}
            placeholder="G"
          />
        </Field>
        <Field label="Seat">
          <input
            className="field-input"
            style={S.input}
            type="number"
            value={seat}
            onChange={(e) => setSeat(e.target.value)}
            placeholder="14"
          />
        </Field>
      </div>

      <Field label="Event Type">
        <select
          className="field-select"
          style={S.input}
          value={eventType}
          onChange={(e) => setEventType(e.target.value)}
        >
          {EVENT_TYPES.map((e) => (
            <option key={e.value} value={e.value}>{e.label}</option>
          ))}
        </select>
      </Field>

      <button
        className="advanced-toggle"
        style={S.advancedToggle}
        onClick={() => setShowAdvanced((v) => !v)}
        type="button"
      >
        <span>Advanced</span>
        <span style={{ transition: 'transform 0.2s ease', transform: showAdvanced ? 'rotate(90deg)' : 'rotate(0deg)' }}>›</span>
      </button>

      <div className={`collapse ${showAdvanced ? 'open' : 'closed'}`}>
        <div style={S.advancedBox}>
          <Field label={`Steps: ${steps}`}>
            <input type="range" min={10} max={50} step={1} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
          </Field>
          <Field label={`Guidance scale: ${guidance.toFixed(1)}`}>
            <input type="range" min={1.0} max={7.0} step={0.1} value={guidance} onChange={(e) => setGuidance(Number(e.target.value))} />
          </Field>
          <Field label="Seed (optional)">
            <input
              className="field-input"
              style={S.input}
              type="number"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="random"
            />
          </Field>
        </div>
      </div>

      <button
        className="primary-btn"
        style={{
          ...S.primaryBtn,
          background: canGenerate ? COLORS.accent : COLORS.borderStrong,
          color: canGenerate ? '#fff' : COLORS.textMuted,
        }}
        disabled={!canGenerate}
        onClick={onGenerate}
        type="button"
      >
        {status === 'offline' ? 'API Offline' : 'Generate View'}
      </button>

      <div style={S.promptPreview}>
        <div style={S.promptLabel}>Prompt</div>
        <div style={S.promptText}>{promptPreview || '—'}</div>
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label style={S.field}>
      <span style={S.fieldLabel}>{label}</span>
      {children}
    </label>
  );
}

function OutputPanel({ genState, progress, result, error, showPromptDetail, setShowPromptDetail, onDownload }) {
  return (
    <div style={S.outputPanel}>
      {genState === 'idle'     && <IdleState />}
      {genState === 'loading'  && <LoadingState progress={progress} />}
      {genState === 'error'    && <ErrorState error={error} />}
      {genState === 'done' && result && (
        <DoneState
          result={result}
          showPromptDetail={showPromptDetail}
          setShowPromptDetail={setShowPromptDetail}
          onDownload={onDownload}
        />
      )}
    </div>
  );
}

function IdleState() {
  return (
    <div style={S.placeholder} className="fade-in">
      <StadiumIcon />
      <div style={S.placeholderTitle}>Configure a seat view</div>
      <div style={S.placeholderSubtitle}>
        Pick a stadium, section, row, and seat — then generate a photorealistic spectator view.
      </div>
    </div>
  );
}

function LoadingState({ progress }) {
  return (
    <div style={S.loadingWrap} className="fade-in">
      <div className="shimmer" style={S.shimmerBlock} />
      <div style={S.loadingFooter}>
        <div style={S.loadingTitle}>
          <span className="spin" style={S.spinner} />
          Generating photorealistic view…
        </div>
        <div style={S.progressOuter}>
          <div style={{ ...S.progressInner, width: `${progress}%` }} />
        </div>
        <div style={S.progressLabel}>{Math.round(progress)}%</div>
      </div>
    </div>
  );
}

function ErrorState({ error }) {
  return (
    <div style={S.placeholder} className="fade-in">
      <div style={{ ...S.placeholderTitle, color: COLORS.error }}>Generation failed</div>
      <div style={S.placeholderSubtitle}>{error}</div>
    </div>
  );
}

function DoneState({ result, showPromptDetail, setShowPromptDetail, onDownload }) {
  return (
    <div style={S.doneWrap} className="fade-in">
      <img
        src={`data:image/jpeg;base64,${result.imageBase64}`}
        alt="Generated seat view"
        style={S.image}
      />
      <div style={S.resultMeta}>
        <div style={S.resultRow}>
          <span className="pill">
            Section {result.section} · Row {result.row} · Seat {result.seat}
          </span>
          <span style={S.timeLabel}>{result.time}s</span>
        </div>
        <button
          className="ghost-btn"
          style={S.ghostBtn}
          onClick={() => setShowPromptDetail((v) => !v)}
          type="button"
        >
          <span style={{ transition: 'transform 0.2s ease', display: 'inline-block', transform: showPromptDetail ? 'rotate(90deg)' : 'rotate(0)', marginRight: 6 }}>›</span>
          Prompt used
        </button>
        <div className={`collapse ${showPromptDetail ? 'open' : 'closed'}`}>
          <div style={S.promptDetail}>{result.prompt}</div>
        </div>
        <button className="primary-btn" style={{ ...S.primaryBtn, background: COLORS.accent, color: '#fff' }} onClick={onDownload} type="button">
          Download Image
        </button>
      </div>
    </div>
  );
}

function StadiumIcon() {
  return (
    <svg width="80" height="80" viewBox="0 0 80 80" style={{ marginBottom: 16 }}>
      <ellipse cx="40" cy="50" rx="32" ry="14" fill="none" stroke={COLORS.borderStrong} strokeWidth="2" />
      <ellipse cx="40" cy="50" rx="22" ry="9"  fill="none" stroke={COLORS.borderStrong} strokeWidth="2" />
      <ellipse cx="40" cy="50" rx="12" ry="5"  fill="none" stroke={COLORS.accent}        strokeWidth="2" />
      <line x1="40" y1="32" x2="40" y2="24" stroke={COLORS.accent} strokeWidth="2" />
      <circle cx="40" cy="22" r="3" fill={COLORS.accent} />
    </svg>
  );
}

const S = {
  app: {
    minHeight: '100vh',
    background: COLORS.bg,
    display: 'flex',
    flexDirection: 'column',
  },
  statsBar: {
    display: 'grid',
    gridTemplateColumns: '1fr auto 1fr',
    alignItems: 'center',
    padding: '14px 28px',
    borderBottom: `1px solid ${COLORS.border}`,
    background: COLORS.surface,
  },
  statsLeft:  { display: 'flex', alignItems: 'center', gap: 10 },
  statsCenter:{ display: 'flex', alignItems: 'center', gap: 14, justifyContent: 'center' },
  statsRight: { display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'flex-end' },
  brandMark:  { color: COLORS.accent, fontSize: 18 },
  brandText:  { fontWeight: 600, letterSpacing: 0.5, color: COLORS.text },
  stat:       { display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 13 },
  statLabel:  { color: COLORS.textDim },
  statValue:  { color: COLORS.text, fontWeight: 500 },
  statSep:    { color: COLORS.textDim },
  statusDot:  { width: 8, height: 8, borderRadius: '50%', display: 'inline-block', transition: 'background 0.3s ease' },

  layout: {
    display: 'grid',
    gridTemplateColumns: '420px 1fr',
    gap: 24,
    padding: 24,
    flex: 1,
    minHeight: 0,
  },

  controlsPanel: {
    background: COLORS.surface,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 12,
    padding: 28,
    display: 'flex',
    flexDirection: 'column',
    gap: 18,
    minWidth: 0,
  },
  title:    { margin: 0, fontSize: 20, fontWeight: 600, color: COLORS.text, letterSpacing: 0.2 },
  subtitle: { margin: '6px 0 0', fontSize: 13, color: COLORS.textMuted },

  field: { display: 'flex', flexDirection: 'column', gap: 6, minWidth: 0 },
  fieldLabel: { fontSize: 12, fontWeight: 500, color: COLORS.textMuted, letterSpacing: 0.4, textTransform: 'uppercase' },
  warnText: { fontSize: 11.5, color: '#f59e0b', marginTop: 4, lineHeight: 1.4 },
  input: {
    background: COLORS.surfaceAlt,
    border: `1px solid ${COLORS.border}`,
    color: COLORS.text,
    borderRadius: 8,
    padding: '10px 12px',
    fontSize: 14,
    width: '100%',
    transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
  },
  row3: { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 },

  advancedToggle: {
    background: 'transparent',
    color: COLORS.textMuted,
    border: 'none',
    padding: 0,
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    fontSize: 13,
    fontWeight: 500,
    alignSelf: 'flex-start',
    transition: 'color 0.2s ease',
  },
  advancedBox: {
    background: COLORS.surfaceAlt,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 8,
    padding: 16,
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
  },

  primaryBtn: {
    width: '100%',
    padding: '13px 16px',
    borderRadius: 8,
    fontSize: 15,
    fontWeight: 600,
    letterSpacing: 0.3,
    transition: 'background 0.2s ease, transform 0.1s ease, color 0.2s ease',
  },

  promptPreview: {
    background: COLORS.surfaceAlt,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 8,
    padding: 12,
  },
  promptLabel: { fontSize: 11, color: COLORS.textDim, fontWeight: 500, textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 6 },
  promptText:  { fontSize: 12.5, color: COLORS.textMuted, lineHeight: 1.55, fontFamily: 'ui-monospace, SF Mono, monospace' },

  outputPanel: {
    background: COLORS.surface,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 12,
    padding: 28,
    display: 'flex',
    flexDirection: 'column',
    minWidth: 0,
    minHeight: 600,
  },
  placeholder: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    textAlign: 'center',
    padding: 32,
  },
  placeholderTitle:    { fontSize: 18, fontWeight: 600, color: COLORS.text, marginBottom: 8 },
  placeholderSubtitle: { fontSize: 14, color: COLORS.textMuted, maxWidth: 340, lineHeight: 1.5 },

  loadingWrap: { flex: 1, display: 'flex', flexDirection: 'column', gap: 16 },
  shimmerBlock: { flex: 1, borderRadius: 10, minHeight: 360 },
  loadingFooter: { display: 'flex', flexDirection: 'column', gap: 8 },
  loadingTitle: { display: 'flex', alignItems: 'center', gap: 10, color: COLORS.text, fontSize: 14, fontWeight: 500 },
  spinner: {
    width: 14, height: 14,
    border: `2px solid ${COLORS.border}`,
    borderTopColor: COLORS.accent,
    borderRadius: '50%',
    display: 'inline-block',
  },
  progressOuter: { width: '100%', height: 4, background: COLORS.border, borderRadius: 999, overflow: 'hidden' },
  progressInner: { height: '100%', background: COLORS.accent, transition: 'width 0.2s ease' },
  progressLabel: { fontSize: 11, color: COLORS.textDim, alignSelf: 'flex-end' },

  doneWrap: { flex: 1, display: 'flex', flexDirection: 'column', gap: 16, minHeight: 0 },
  image: {
    width: '100%',
    maxHeight: 'calc(100vh - 360px)',
    objectFit: 'contain',
    borderRadius: 10,
    background: COLORS.bg,
  },
  resultMeta: { display: 'flex', flexDirection: 'column', gap: 12 },
  resultRow: { display: 'flex', alignItems: 'center', justifyContent: 'space-between' },
  timeLabel: { fontSize: 12, color: COLORS.textMuted, fontFamily: 'ui-monospace, SF Mono, monospace' },
  ghostBtn: {
    background: 'transparent',
    color: COLORS.textMuted,
    padding: '8px 0',
    fontSize: 13,
    fontWeight: 500,
    textAlign: 'left',
    transition: 'color 0.2s ease, background 0.2s ease',
    borderRadius: 6,
  },
  promptDetail: {
    background: COLORS.surfaceAlt,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 8,
    padding: 12,
    fontSize: 12.5,
    color: COLORS.textMuted,
    lineHeight: 1.55,
    fontFamily: 'ui-monospace, SF Mono, monospace',
    marginBottom: 4,
  },
};
