import { useEffect, useRef, useState } from "react";
import { api } from "./lib/api";
import { CompareModesPanel } from "./components/CompareModesPanel";
import { ControlPanel } from "./components/ControlPanel";
import { DecisionPanel } from "./components/DecisionPanel";
import { EventBanner } from "./components/EventBanner";
import { Header } from "./components/Header";
import { MetricsPanel } from "./components/MetricsPanel";
import { PatientPanel } from "./components/PatientPanel";
import { ResourcePanel } from "./components/ResourcePanel";

const defaultOptions = {
  tasks: [],
  ethicalModes: [],
};

const defaultControls = {
  difficulty: "medium",
  ethical_mode: "fairness",
  num_agents: 3,
  uncertainty: true,
  dynamic_events: true,
};

export default function App() {
  const [options, setOptions] = useState(defaultOptions);
  const [controls, setControls] = useState(defaultControls);
  const [dashboard, setDashboard] = useState(null);
  const [comparison, setComparison] = useState([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [compareBusy, setCompareBusy] = useState(false);
  const [error, setError] = useState("");
  const [isAutoRunning, setIsAutoRunning] = useState(false);
  const runLoopRef = useRef(null);

  useEffect(() => {
    let mounted = true;

    async function bootstrap() {
      try {
        const config = await api.getConfig();
        if (!mounted) {
          return;
        }
        const nextControls = {
          ...defaultControls,
          ...config.defaults,
        };
        setOptions({
          tasks: config.tasks || [],
          ethicalModes: config.ethical_modes || [],
        });
        setControls(nextControls);
        const snapshot = await api.reset(nextControls);
        if (!mounted) {
          return;
        }
        setDashboard(snapshot);
        const compare = await api.compare(nextControls);
        if (!mounted) {
          return;
        }
        setComparison(compare.comparison || []);
      } catch (bootstrapError) {
        if (mounted) {
          setError(bootstrapError.message);
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    }

    bootstrap();

    return () => {
      mounted = false;
      if (runLoopRef.current) {
        window.clearTimeout(runLoopRef.current);
      }
    };
  }, []);

  async function refreshCompare(nextControls = controls) {
    setCompareBusy(true);
    try {
      const compare = await api.compare(nextControls);
      setComparison(compare.comparison || []);
    } catch (compareError) {
      setError(compareError.message);
    } finally {
      setCompareBusy(false);
    }
  }

  async function handleReset() {
    handlePause();
    setBusy(true);
    setError("");
    try {
      const snapshot = await api.reset(controls);
      setDashboard(snapshot);
      await refreshCompare(controls);
    } catch (resetError) {
      setError(resetError.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleStep() {
    setBusy(true);
    setError("");
    try {
      const snapshot = await api.step(1);
      setDashboard(snapshot);
    } catch (stepError) {
      setError(stepError.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleRunFull() {
    if (isAutoRunning) {
      return;
    }
    setIsAutoRunning(true);
    setError("");

    const runTick = async () => {
      try {
        const snapshot = await api.step(1);
        setDashboard(snapshot);
        if (!snapshot.status.done) {
          runLoopRef.current = window.setTimeout(runTick, 450);
        } else {
          setIsAutoRunning(false);
        }
      } catch (runError) {
        setError(runError.message);
        setIsAutoRunning(false);
      }
    };

    runTick();
  }

  function handlePause() {
    if (runLoopRef.current) {
      window.clearTimeout(runLoopRef.current);
    }
    runLoopRef.current = null;
    setIsAutoRunning(false);
  }

  function handleControlChange(key, value) {
    setControls((current) => ({
      ...current,
      [key]: value,
    }));
  }

  if (loading) {
    return (
      <main className="min-h-screen px-4 py-8 sm:px-6 lg:px-8">
        <div className="mx-auto grid min-h-[80vh] max-w-7xl place-items-center rounded-[32px] border border-white/70 bg-white/80 p-10 shadow-panel">
          <div className="text-center">
            <div className="mx-auto h-16 w-16 animate-pulse-soft rounded-3xl bg-clinical-500/20 p-4">
              <div className="h-full w-full rounded-2xl bg-clinical-500" />
            </div>
            <p className="mt-5 font-display text-2xl text-ink">Booting command center...</p>
          </div>
        </div>
      </main>
    );
  }

  const state = dashboard?.state || {};
  const metrics = dashboard?.metrics || {};
  const latestInfo = dashboard?.latest_step?.info || {};
  const activeEvents = dashboard?.observation?.active_events || [];

  return (
    <main className="min-h-screen px-4 py-5 sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-5 pb-10">
        <Header
          summary={metrics.summary}
          status={dashboard?.status}
          config={dashboard?.config}
        />

        <EventBanner events={activeEvents} />

        {error ? (
          <div className="rounded-[24px] border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700">
            {error}
          </div>
        ) : null}

        <ControlPanel
          controls={controls}
          options={options}
          busy={busy}
          isAutoRunning={isAutoRunning}
          onChange={handleControlChange}
          onReset={handleReset}
          onStep={handleStep}
          onRunFull={handleRunFull}
          onPause={handlePause}
        />

        <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
          <PatientPanel patients={state.queue?.waiting || []} />
          <ResourcePanel resources={state.resources || {}} />
        </div>

        <div className="grid gap-5 xl:grid-cols-[0.9fr_1.1fr]">
          <DecisionPanel
            actions={latestInfo.actions_per_agent || []}
            conflicts={latestInfo.conflicts || []}
            reasoning={latestInfo.decision_reasoning}
            ethicalMode={dashboard?.config?.ethical_mode || "fairness"}
          />
          <MetricsPanel history={metrics.history || []} summary={metrics.summary || {}} />
        </div>

        <CompareModesPanel
          data={comparison}
          busy={compareBusy}
          onRefresh={() => refreshCompare()}
        />
      </div>
    </main>
  );
}
