import React, { useState, useEffect, useRef } from 'react';

// Known FIFA 2022 WC teams for the dropdown
const WC2022_TEAMS = [
  "Argentina","Australia","Austria","Belgium","Brazil","Cameroon","Canada",
  "Costa Rica","Croatia","Denmark","Ecuador","England","France",
  "Germany","Ghana","Iran","Japan","Mexico","Morocco","Netherlands",
  "Poland","Portugal","Qatar","Saudi Arabia","Senegal","Serbia",
  "South Korea","Spain","Switzerland","Tunisia","United States",
  "Uruguay","Wales"
];

// Team default kit colors (primary jersey)
const TEAM_COLORS = {
  "Argentina": "#74ACDF", "Australia": "#FFCD00", "Austria": "#E21E26", "Belgium": "#EF3340",
  "Brazil": "#F7D116", "Cameroon": "#007A5E", "Canada": "#FF0000",
  "Costa Rica": "#002B7F", "Croatia": "#FF0000", "Denmark": "#C60C30",
  "Ecuador": "#FFD100", "England": "#FFFFFF", "France": "#003189",
  "Germany": "#FFFFFF", "Ghana": "#006B3F", "Iran": "#239F40",
  "Japan": "#003087", "Mexico": "#006847", "Morocco": "#C1272D",
  "Netherlands": "#FF6600", "Poland": "#FFFFFF", "Portugal": "#006600",
  "Qatar": "#8D1B3D", "Saudi Arabia": "#006C35", "Senegal": "#00853F",
  "Serbia": "#C6363C", "South Korea": "#CD2E3A", "Spain": "#AA151B",
  "Switzerland": "#FF0000", "Tunisia": "#E70013", "United States": "#002868",
  "Uruguay": "#5EB6E4", "Wales": "#C8102E"
};

function App() {
  // Step management: 0=match context, 1=params, 2=upload
  const [step, setStep] = useState(0);

  // Match Context
  const [teamA, setTeamA] = useState('Spain');
  const [teamB, setTeamB] = useState('Austria');
  const [teamAColor, setTeamAColor] = useState(TEAM_COLORS['Spain']);
  const [teamBColor, setTeamBColor] = useState(TEAM_COLORS['Austria']);

  // Pipeline params
  const [sceneThreshold, setSceneThreshold] = useState(27.0);
  const [conf, setConf] = useState(0.25);
  const [useKalman, setUseKalman] = useState(true);
  const [file, setFile] = useState(null);

  const [isDragOver, setIsDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  
  const [processingState, setProcessingState] = useState({
    status: 'idle', percent: 0, message: 'Ready to upload.',
    original_filename: '', processed_filename: '', error: ''
  });

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // Real-time analytics telemetry
  const [telemetry, setTelemetry] = useState(null);

  const originalVideoRef = useRef(null);
  const gamifiedVideoRef = useRef(null);
  const pollingIntervalRef = useRef(null);

  // Auto-update color when team changes
  useEffect(() => {
    if (TEAM_COLORS[teamA]) setTeamAColor(TEAM_COLORS[teamA]);
  }, [teamA]);
  useEffect(() => {
    if (TEAM_COLORS[teamB]) setTeamBColor(TEAM_COLORS[teamB]);
  }, [teamB]);

  // Load telemetry when processing completes
  useEffect(() => {
    if (processingState.status === 'completed' && processingState.processed_filename) {
      fetchTelemetry(processingState.processed_filename);
    }
  }, [processingState.status, processingState.processed_filename]);

  const fetchTelemetry = async (filename) => {
    try {
      const res = await fetch(`/videos/processed/${filename}.json`);
      if (res.ok) {
        const data = await res.json();
        setTelemetry(data);
        console.log("Loaded telemetry metadata successfully:", data);
      }
    } catch (err) {
      console.error("Error loading telemetry JSON:", err);
    }
  };

  useEffect(() => {
    if (processingState.status === 'processing') {
      pollingIntervalRef.current = setInterval(checkProgress, 500);
    } else {
      if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current);
    }
    return () => { if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current); };
  }, [processingState.status]);

  const checkProgress = async () => {
    try {
      const res = await fetch('/api/progress');
      const data = await res.json();
      setProcessingState(data);
    } catch (err) { console.error('Error polling progress:', err); }
  };

  const handleDragOver = (e) => { e.preventDefault(); setIsDragOver(true); };
  const handleDragLeave = () => setIsDragOver(false);
  const handleDrop = (e) => {
    e.preventDefault(); setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) validateAndSetFile(files[0]);
  };
  const handleFileChange = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) validateAndSetFile(files[0]);
  };
  const validateAndSetFile = (selectedFile) => {
    if (!selectedFile.type.startsWith('video/')) { alert('Invalid format: select an MP4 video.'); return; }
    if (selectedFile.size > 150 * 1024 * 1024) { alert('File too large (max 150MB).'); return; }
    setFile(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('scene_threshold', sceneThreshold);
    formData.append('conf', conf);
    formData.append('use_kalman', useKalman);
    formData.append('team_a_name', teamA.toLowerCase());
    formData.append('team_b_name', teamB.toLowerCase());
    formData.append('team_a_color', teamAColor);
    formData.append('team_b_color', teamBColor);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload', true);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setUploadProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = function () {
      setLoading(false);
      if (xhr.status === 200 || xhr.status === 201) {
        setProcessingState({ status: 'processing', percent: 0, message: 'Video uploaded. Running shot segmentation...', original_filename: '', processed_filename: '', error: '' });
      } else {
        const err = JSON.parse(xhr.responseText);
        alert(`Upload failed: ${err.error || xhr.statusText}`);
      }
    };
    xhr.onerror = function () { setLoading(false); alert('Upload error: check connection to server.'); };
    xhr.send(formData);
  };

  const handleReset = async () => {
    try {
      const res = await fetch('/api/reset', { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setProcessingState(data); setFile(null); setTelemetry(null);
        setIsPlaying(false); setCurrentTime(0); setDuration(0);
        setStep(0);
      } else { alert(data.error); }
    } catch (err) { console.error('Error resetting:', err); }
  };

  const handlePlayPause = () => {
    const og = originalVideoRef.current;
    const gm = gamifiedVideoRef.current;
    if (!og || !gm) return;
    if (isPlaying) { og.pause(); gm.pause(); setIsPlaying(false); }
    else { og.currentTime = gm.currentTime; og.play().catch(()=>{}); gm.play().catch(()=>{}); setIsPlaying(true); }
  };

  const handleTimeUpdate = () => { if (gamifiedVideoRef.current) setCurrentTime(gamifiedVideoRef.current.currentTime); };
  const handleLoadedMetadata = () => { if (gamifiedVideoRef.current) setDuration(gamifiedVideoRef.current.duration); };
  const handleTimelineChange = (e) => {
    const t = parseFloat(e.target.value);
    if (originalVideoRef.current) originalVideoRef.current.currentTime = t;
    if (gamifiedVideoRef.current) gamifiedVideoRef.current.currentTime = t;
    setCurrentTime(t);
  };

  useEffect(() => {
    const interval = setInterval(() => {
      const og = originalVideoRef.current;
      const gm = gamifiedVideoRef.current;
      if (og && gm && isPlaying && Math.abs(og.currentTime - gm.currentTime) > 0.15) og.currentTime = gm.currentTime;
    }, 1000);
    return () => clearInterval(interval);
  }, [isPlaying]);

  const originalSrc = processingState.original_filename ? `/videos/original/${processingState.original_filename}` : null;
  const gamifiedSrc = processingState.processed_filename ? `/videos/processed/${processingState.processed_filename}` : null;

  const formatTime = (t) => {
    if (isNaN(t)) return '00:00';
    const m = Math.floor(t / 60), s = Math.floor(t % 60);
    return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
  };

  // Resolve current frame index and its telemetry
  const fps = telemetry?.fps || 30.0;
  const totalFrames = telemetry?.total_frames || 1;
  const currentFrameIdx = Math.min(Math.round(currentTime * fps), totalFrames - 1);
  const currentFrameData = telemetry?.timeline?.[currentFrameIdx] || {
    ball_speed: 0.0, possession_pct_a: 50.0, possession_pct_b: 50.0, camera_motion: 0.0, players: []
  };

  // Separate players into their respective team slots
  const playersA = currentFrameData.players.filter(p => p.team_id === 0);
  const playersB = currentFrameData.players.filter(p => p.team_id === 1);

  // Compute camera panning direction text
  const getCameraPanningText = (motion) => {
    if (motion < 0.8) return "Camera Static";
    if (motion < 4.0) return "Slow Camera Pan";
    return "Fast Camera Pan";
  };

  // ─── STEP INDICATOR ────────────────────────────────────────────────────────
  const StepIndicator = () => (
    <div className="step-indicator">
      {['Match Context','Parameters','Upload'].map((label, i) => (
        <div key={i} className={`step-dot-wrap ${i <= step ? 'active' : ''}`}>
          <div className={`step-dot ${i < step ? 'done' : i === step ? 'current' : ''}`}>
            {i < step ? '✓' : i + 1}
          </div>
          <span className="step-label">{label}</span>
          {i < 2 && <div className={`step-line ${i < step ? 'done' : ''}`} />}
        </div>
      ))}
    </div>
  );

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div className="logo-section">
          <h1>GameCast <span className="logo-badge">HUD</span></h1>
          <p>Gamified Football Highlight Video Overlay Pipeline</p>
        </div>
      </header>

      {/* ── IDLE SETUP FLOW ─────────────────────────────────────────────── */}
      {processingState.status === 'idle' && (
        <>
          <StepIndicator />

          {/* Step 0: Match Context */}
          {step === 0 && (
            <section className="glass-panel">
              <h2 className="panel-title">⚽ Match Context</h2>
              <p className="panel-subtitle">Tell GameCast which teams are playing so it can show real player names.</p>

              <div className="team-context-grid">
                {/* Team A */}
                <div className="team-context-card team-a" style={{ borderColor: teamAColor }}>
                  <div className="team-badge" style={{ background: teamAColor }}>A</div>
                  <div className="team-form">
                    <label className="form-label">Team A</label>
                    <select
                      className="form-select"
                      value={teamA}
                      onChange={e => setTeamA(e.target.value)}
                    >
                      {WC2022_TEAMS.map(t => <option key={t} value={t}>{t}</option>)}
                    </select>
                    <label className="form-label" style={{ marginTop: '0.75rem' }}>Kit Color</label>
                    <div className="color-row">
                      <input
                        type="color"
                        className="color-picker"
                        value={teamAColor}
                        onChange={e => setTeamAColor(e.target.value)}
                      />
                      <span className="color-hex">{teamAColor.toUpperCase()}</span>
                    </div>
                  </div>
                </div>

                <div className="vs-divider">VS</div>

                {/* Team B */}
                <div className="team-context-card team-b" style={{ borderColor: teamBColor }}>
                  <div className="team-badge" style={{ background: teamBColor }}>B</div>
                  <div className="team-form">
                    <label className="form-label">Team B</label>
                    <select
                      className="form-select"
                      value={teamB}
                      onChange={e => setTeamB(e.target.value)}
                    >
                      {WC2022_TEAMS.map(t => <option key={t} value={t}>{t}</option>)}
                    </select>
                    <label className="form-label" style={{ marginTop: '0.75rem' }}>Kit Color</label>
                    <div className="color-row">
                      <input
                        type="color"
                        className="color-picker"
                        value={teamBColor}
                        onChange={e => setTeamBColor(e.target.value)}
                      />
                      <span className="color-hex">{teamBColor.toUpperCase()}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Match preview strip */}
              <div className="match-preview-strip">
                <div className="match-team-chip" style={{ background: teamAColor, color: '#000' }}>
                  {teamA.toUpperCase()}
                </div>
                <span className="match-preview-vs">⚽ FIFA World Cup 2022</span>
                <div className="match-team-chip" style={{ background: teamBColor, color: '#000' }}>
                  {teamB.toUpperCase()}
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1.5rem' }}>
                <button className="upload-btn" onClick={() => setStep(1)}>
                  Next: Parameters →
                </button>
              </div>
            </section>
          )}

          {/* Step 1: Parameters */}
          {step === 1 && (
            <section className="glass-panel">
              <h2 className="panel-title">⚙️ Pipeline Parameters</h2>
              <div className="config-grid">
                <div className="config-item">
                  <label>Scene Cut Threshold<span className="config-value">{sceneThreshold}</span></label>
                  <input type="range" min="10" max="50" step="0.5" value={sceneThreshold}
                    onChange={e => setSceneThreshold(parseFloat(e.target.value))} className="config-slider" />
                </div>
                <div className="config-item">
                  <label>YOLO Confidence<span className="config-value">{conf}</span></label>
                  <input type="range" min="0.10" max="0.90" step="0.05" value={conf}
                    onChange={e => setConf(parseFloat(e.target.value))} className="config-slider" />
                </div>
                <div className="config-item">
                  <label className="config-checkbox-container">
                    <input type="checkbox" checked={useKalman} onChange={e => setUseKalman(e.target.checked)} />
                    <span className="custom-checkbox"></span>
                    <span className="config-label-text">Kalman Smoothing</span>
                  </label>
                </div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1.5rem' }}>
                <button className="back-btn" onClick={() => setStep(0)}>← Back</button>
                <button className="upload-btn" onClick={() => setStep(2)}>Next: Upload →</button>
              </div>
            </section>
          )}

          {/* Step 2: Upload */}
          {step === 2 && (
            <section className="glass-panel">
              <h2 className="panel-title">📁 Upload Highlight Video</h2>

              {/* Match reminder */}
              <div className="match-reminder">
                <div className="reminder-chip" style={{ background: teamAColor }}>
                  {teamA}
                </div>
                <span>vs</span>
                <div className="reminder-chip" style={{ background: teamBColor }}>
                  {teamB}
                </div>
                <button className="edit-match-btn" onClick={() => setStep(0)}>✏️ Change</button>
              </div>

              <div
                className={`upload-zone ${isDragOver ? 'dragover' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-picker').click()}
              >
                <div className="upload-icon">⇪</div>
                <input id="file-picker" type="file" accept="video/*" onChange={handleFileChange} className="file-input" />
                {file ? (
                  <div>
                    <h3 style={{ color: 'var(--color-cyan)' }}>{file.name}</h3>
                    <p>{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                  </div>
                ) : (
                  <div>
                    <h3>Drag & Drop FIFA highlight video</h3>
                    <p>Or click to select a file (MP4, max 150MB)</p>
                  </div>
                )}
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1.5rem' }}>
                <button className="back-btn" onClick={() => setStep(1)}>← Back</button>
                {file && (
                  <button className="upload-btn" onClick={handleUpload} disabled={loading}>
                    {loading ? `Uploading (${uploadProgress}%)` : '🚀 Start Gamification'}
                  </button>
                )}
              </div>
            </section>
          )}
        </>
      )}

      {/* ── PROCESSING ─────────────────────────────────────────────────────── */}
      {processingState.status === 'processing' && (
        <section className="glass-panel processing-card">
          <div className="processing-teams">
            <div className="proc-team" style={{ color: teamAColor }}>{teamA}</div>
            <div className="proc-vs">⚽</div>
            <div className="proc-team" style={{ color: teamBColor }}>{teamB}</div>
          </div>
          <div className="progress-header">
            <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', color: 'var(--color-cyan)' }}>
              Gamifying Highlight Clip...
            </h2>
            <div className="status-badge">Processing</div>
          </div>
          <div className="progress-bar-bg">
            <div className="progress-bar-fill" style={{ width: `${processingState.percent}%` }} />
          </div>
          <div className="status-ticker">{processingState.message}</div>
        </section>
      )}

      {/* ── COMPLETED ──────────────────────────────────────────────────────── */}
      {processingState.status === 'completed' && (
        <>
          <section className="video-compare-section">
            <div className="compare-header">
              <div>
                <h2>Dual Sync Playback</h2>
                <div className="match-reminder" style={{ marginTop: '0.5rem' }}>
                  <div className="reminder-chip" style={{ background: teamAColor }}>{teamA}</div>
                  <span style={{ color: 'var(--color-text-muted)', fontSize: '0.8rem' }}>vs</span>
                  <div className="reminder-chip" style={{ background: teamBColor }}>{teamB}</div>
                </div>
              </div>
              <button className="reset-button" onClick={handleReset}>Upload Another</button>
            </div>

            <div className="video-grid">
              <div className="video-card">
                <div className="video-card-title"><span className="dot original"></span> Original Broadcast</div>
                <div className="video-wrapper">
                  <video ref={originalVideoRef} src={originalSrc} className="video-element" muted playsInline />
                </div>
              </div>
              <div className="video-card">
                <div className="video-card-title"><span className="dot gamified"></span> GameCast HUD Overlay</div>
                <div className="video-wrapper">
                  <video ref={gamifiedVideoRef} src={gamifiedSrc} className="video-element" playsInline
                    onTimeUpdate={handleTimeUpdate} onLoadedMetadata={handleLoadedMetadata}
                    onEnded={() => setIsPlaying(false)} />
                </div>
              </div>
            </div>

            <div className="sync-controls-panel">
              <button className="control-btn play-pause" onClick={handlePlayPause}>
                {isPlaying ? '⏸' : '▶'}
              </button>
              <div className="timeline-slider-container">
                <span className="time-display">{formatTime(currentTime)}</span>
                <input type="range" min="0" max={duration || 0} step="0.05"
                  value={currentTime} onChange={handleTimelineChange}
                  className="config-slider" style={{ flexGrow: 1 }} />
                <span className="time-display">{formatTime(duration)}</span>
              </div>
            </div>

            <div className="effects-info-panel">
              <div className="effect-badge player"><span className="dot"></span> Team-Colored Player Boxes</div>
              <div className="effect-badge player" style={{ '--dot-color': teamAColor }}>
                <span className="dot" style={{ background: teamAColor }}></span> {teamA} Players
              </div>
              <div className="effect-badge player" style={{ '--dot-color': teamBColor }}>
                <span className="dot" style={{ background: teamBColor }}></span> {teamB} Players
              </div>
              <div className="effect-badge ball"><span className="dot"></span> Glowing Ball Trail</div>
              <div className="effect-badge impact"><span className="dot"></span> Kick Shockwaves</div>
              <div className="effect-badge impact" style={{ '--dot-color': '#ff3366' }}>
                <span className="dot" style={{ background: '#ff3366' }}></span> Danger Zone Alert
              </div>
            </div>
          </section>

          {/* ── DYNAMIC ANALYTICS DASHBOARD ─────────────────────────────────── */}
          <section className="glass-panel telemetry-dashboard-panel" style={{ marginTop: '1.5rem' }}>
            <h2 className="panel-title">📊 Dynamic Match Telemetry</h2>
            <p className="panel-subtitle">Real-time match data synchronized frame-by-frame with the highlight video</p>

            <div className="telemetry-widgets-grid">
              {/* Ball Speed widget */}
              <div className="telemetry-card speed-card">
                <div className="card-header">
                  <span className="card-icon">⚡</span>
                  <span className="card-title">Ball Velocity</span>
                </div>
                <div className="card-value-display">
                  <span className="big-value">{currentFrameData.ball_speed}</span>
                  <span className="value-unit">KM/H</span>
                </div>
                <div className="speed-progress-bg">
                  <div className="speed-progress-fill" style={{ width: `${Math.min(100, (currentFrameData.ball_speed / 130.0) * 100)}%` }} />
                </div>
              </div>

              {/* Possession control widget */}
              <div className="telemetry-card possession-card">
                <div className="card-header">
                  <span className="card-icon">⚔️</span>
                  <span className="card-title">Possession Control</span>
                </div>
                <div className="possession-matchup-row">
                  <div className="poss-team-percent team-a" style={{ color: teamAColor }}>
                    <div className="poss-pct">{currentFrameData.possession_pct_a}%</div>
                    <div className="poss-name">{teamA}</div>
                  </div>
                  <div className="poss-vs-divider">/</div>
                  <div className="poss-team-percent team-b" style={{ color: teamBColor }}>
                    <div className="poss-pct">{currentFrameData.possession_pct_b}%</div>
                    <div className="poss-name">{teamB}</div>
                  </div>
                </div>
                <div className="possession-ratio-bar">
                  <div className="ratio-fill team-a" style={{ width: `${currentFrameData.possession_pct_a}%`, background: teamAColor }} />
                  <div className="ratio-fill team-b" style={{ width: `${currentFrameData.possession_pct_b}%`, background: teamBColor }} />
                </div>
              </div>

              {/* Camera Movement widget */}
              <div className="telemetry-card camera-card">
                <div className="card-header">
                  <span className="card-icon">📹</span>
                  <span className="card-title">Camera movement</span>
                </div>
                <div className="card-value-display">
                  <span className="big-value">{currentFrameData.camera_motion.toFixed(1)}</span>
                  <span className="value-unit">PX/F</span>
                </div>
                <div className="camera-state-text" style={{ 
                  color: currentFrameData.camera_motion >= 0.8 ? 'var(--color-cyan)' : 'var(--color-text-muted)' 
                }}>
                  {getCameraPanningText(currentFrameData.camera_motion)}
                </div>
              </div>
            </div>

            {/* Player Performance stats table */}
            <div className="player-stats-section">
              <h3 className="section-subtitle">🏃‍♂️ Tracked Player Telemetry</h3>
              <div className="player-roster-split-grid">
                {/* Team A Players */}
                <div className="team-roster-card" style={{ borderColor: teamAColor }}>
                  <h4 className="roster-team-title" style={{ color: teamAColor }}>{teamA} squad</h4>
                  <div className="roster-list">
                    {playersA.length === 0 ? (
                      <p className="no-players-text">No Spain players in tracking zone</p>
                    ) : (
                      playersA.map(p => (
                        <div key={p.track_id} className="player-row-item">
                          <span className="player-name-lbl">🏃 {p.name}</span>
                          <div className="player-stat-details">
                            <span className="stat-chip speed">{p.speed_kmh} km/h</span>
                            <span className="stat-chip dist">{p.distance_m} m</span>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>

                {/* Team B Players */}
                <div className="team-roster-card" style={{ borderColor: teamBColor }}>
                  <h4 className="roster-team-title" style={{ color: teamBColor }}>{teamB} squad</h4>
                  <div className="roster-list">
                    {playersB.length === 0 ? (
                      <p className="no-players-text">No {teamB} players in tracking zone</p>
                    ) : (
                      playersB.map(p => (
                        <div key={p.track_id} className="player-row-item">
                          <span className="player-name-lbl">🏃 {p.name}</span>
                          <div className="player-stat-details">
                            <span className="stat-chip speed">{p.speed_kmh} km/h</span>
                            <span className="stat-chip dist">{p.distance_m} m</span>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </div>
          </section>
        </>
      )}

      {/* ── FAILED ─────────────────────────────────────────────────────────── */}
      {processingState.status === 'failed' && (
        <section className="glass-panel error-card">
          <div className="error-header">Pipeline Error Encountered</div>
          <p style={{ color: '#ff3366', fontSize: '0.95rem', marginBottom: '1.5rem' }}>
            {processingState.message}
          </p>
          <button className="reset-button" onClick={handleReset}>Go Back</button>
        </section>
      )}
    </div>
  );
}

export default App;
