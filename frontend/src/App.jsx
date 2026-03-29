import { useState } from "react";

const API_BASE = "http://localhost:9000";

export default function App() {
  const [tab, setTab] = useState("generator");

  // Generator state
  const [topic, setTopic] = useState("transfer case");
  const [difficulty, setDifficulty] = useState("medium");
  const [numQuestions, setNumQuestions] = useState(5);
  const [types, setTypes] = useState(["mcq"]);
  const [paper, setPaper] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Custom prompt
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState(null);

  const toggleType = (type) => {
    setTypes((prev) =>
      prev.includes(type)
        ? prev.filter((t) => t !== type)
        : [...prev, type]
    );
  };

  // 🔥 GENERATE PAPER
  const generatePaper = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/generate-paper`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic,
          difficulty,
          num_questions: numQuestions,
          types,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "API error");
      }

      setPaper(data.paper);
      setTab("preview");
    } catch (err) {
      console.error(err);
      setError(err.message);
    }
    setLoading(false);
  };

  // 🔥 DOWNLOAD
  const download = async (type) => {
    try {
      const res = await fetch(`${API_BASE}/download/${type}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paper, topic, difficulty }),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt);
      }

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = url;
      a.download = `paper.${type}`;
      a.click();
    } catch (err) {
      alert("Download failed: " + err.message);
    }
  };

  // 🔥 CUSTOM PROMPT
  const runPrompt = async () => {
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: prompt }),
      });

      const data = await res.json();
      setResponse(data);
    } catch (err) {
      alert("Error running prompt");
    }
  };

  return (
    <div style={{ padding: 30, fontFamily: "Arial" }}>
      <h1>AI Question Paper System</h1>

      {/* Tabs */}
      <div style={{ marginBottom: 20 }}>
        <button onClick={() => setTab("generator")}>Generator</button>
        <button onClick={() => setTab("preview")}>Preview</button>
        <button onClick={() => setTab("custom")}>Custom Prompt</button>
      </div>

      {/* ================= GENERATOR ================= */}
      {tab === "generator" && (
        <div>
          <h2>Generate Paper</h2>

          <input
            placeholder="Topic"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
          />

          <br /><br />

          <select
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value)}
          >
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
          </select>

          <br /><br />

          <input
            type="number"
            value={numQuestions}
            onChange={(e) => setNumQuestions(Number(e.target.value))}
          />

          <br /><br />

          <div>
            {["mcq", "short", "long"].map((t) => (
              <label key={t} style={{ marginRight: 10 }}>
                <input
                  type="checkbox"
                  checked={types.includes(t)}
                  onChange={() => toggleType(t)}
                />
                {t}
              </label>
            ))}
          </div>

          <br />

          <button onClick={generatePaper}>
            {loading ? "Generating..." : "Generate"}
          </button>

          {error && (
            <p style={{ color: "red" }}>
              ❌ {error}
            </p>
          )}
        </div>
      )}

      {/* ================= PREVIEW ================= */}
      {tab === "preview" && (
        <div>
          <h2>Preview</h2>

          {!paper && <p>No paper generated</p>}

          {paper?.mcq?.map((q, i) => (
            <div key={i}>
              <p>{i + 1}. {q.question}</p>
            </div>
          ))}

          <br />

          <button onClick={() => download("docx")}>Download DOCX</button>
          <button onClick={() => download("pdf")}>Download PDF</button>
        </div>
      )}

      {/* ================= CUSTOM ================= */}
      {tab === "custom" && (
        <div>
          <h2>Custom Prompt</h2>

          <textarea
            rows={5}
            cols={60}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />

          <br /><br />

          <button onClick={runPrompt}>Run</button>

          {response && (
            <pre style={{ marginTop: 20 }}>
              {JSON.stringify(response, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}