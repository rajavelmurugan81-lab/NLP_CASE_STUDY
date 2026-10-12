/**
 * Frontend JavaScript for Academic Literature Review Dashboard
 * Project: Agentic AI for Automated Literature Review Using LLMs and RAG
 * Course: CSI4001 NLP and Computational Linguistics
 */

document.addEventListener("DOMContentLoaded", () => {
    // Health & System Elements
    const statusSpinner = document.getElementById("status-spinner");
    const statusIcon = document.getElementById("status-icon");
    const statusText = document.getElementById("status-text");
    const statusCard = document.querySelector(".status-card");
    const recheckBtn = document.getElementById("btn-recheck-health");
    const kbStatsBadge = document.getElementById("kb-stats-badge");

    // Search Elements
    const searchForm = document.getElementById("search-form");
    const topicInput = document.getElementById("topic-input");
    const limitSelect = document.getElementById("limit-select");
    const searchAlert = document.getElementById("search-alert");
    const searchBtn = document.getElementById("btn-search");
    const resultsSection = document.getElementById("results-section");
    const resultsTitle = document.getElementById("results-title");
    const resultsSubtitle = document.getElementById("results-subtitle");
    const papersList = document.getElementById("papers-list");

    // RAG Upload & Ingestion Elements
    const pdfFilesInput = document.getElementById("pdf-files-input");
    const uploadBtn = document.getElementById("btn-upload");
    const ingestBtn = document.getElementById("btn-ingest");
    const uploadAlert = document.getElementById("upload-alert");
    const uploadedFilesBox = document.getElementById("uploaded-files-box");
    const uploadedFilenameBadge = document.getElementById("uploaded-filename-badge");
    let currentUploadedFilename = null;

    // RAG Retrieval Elements
    const retrieveForm = document.getElementById("retrieve-form");
    const ragQueryInput = document.getElementById("rag-query-input");
    const topKSelect = document.getElementById("top-k-select");
    const retrievalModeSelect = document.getElementById("retrieval-mode-select");
    const retrievalModeBadge = document.getElementById("retrieval-mode-badge");
    const retrieveBtn = document.getElementById("btn-retrieve");
    const retrieveAlert = document.getElementById("retrieve-alert");
    const passagesSection = document.getElementById("passages-section");
    const passagesTitle = document.getElementById("passages-title");
    const passagesSubtitle = document.getElementById("passages-subtitle");
    const passagesList = document.getElementById("passages-list");

    // Agentic Workflow Elements
    const wfLlmBadge = document.getElementById("wf-llm-badge");
    const wfForm = document.getElementById("workflow-form");
    const wfQuestionInput = document.getElementById("wf-question-input");
    const wfSubtopicsInput = document.getElementById("wf-subtopics-input");
    const wfPapersInput = document.getElementById("wf-papers-input");
    const wfRetriesInput = document.getElementById("wf-retries-input");
    const btnRunWorkflow = document.getElementById("btn-run-workflow");
    const wfAlert = document.getElementById("wf-alert");
    const wfProgressCard = document.getElementById("wf-progress-card");
    const wfSpinner = document.getElementById("wf-spinner");
    const wfStatusMessage = document.getElementById("wf-status-message");
    const wfRetryBadge = document.getElementById("wf-retry-badge");
    const wfResultsContainer = document.getElementById("wf-results-container");
    const wfSummaryMeta = document.getElementById("wf-summary-meta");
    const btnCopyReport = document.getElementById("btn-copy-report");
    const wfReportTitle = document.getElementById("wf-report-title");
    const wfReportStats = document.getElementById("wf-report-stats");
    const wfReportRendered = document.getElementById("wf-report-rendered");
    const wfMatrixTbody = document.getElementById("wf-matrix-tbody");
    const wfClustersContainer = document.getElementById("wf-clusters-container");
    const wfGapsContainer = document.getElementById("wf-gaps-container");
    const wfVerificationContainer = document.getElementById("wf-verification-container");
    const wfPlanContainer = document.getElementById("wf-plan-container");
    let latestReportMarkdown = "";

    /**
     * Checks backend health via GET /api/health and updates KB stats
     */
    async function checkBackendHealth() {
        if (!statusText) return;

        statusSpinner.classList.remove("d-none");
        statusIcon.classList.add("d-none");
        statusText.textContent = "Connecting to /api/health...";
        statusCard.classList.remove("status-healthy", "status-error");

        try {
            const response = await fetch("/api/health");
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const data = await response.json();

            statusSpinner.classList.add("d-none");
            statusIcon.className = "bi bi-check-circle-fill text-success fs-5";
            statusText.textContent = `Online (${data.status.toUpperCase()} - ${data.service})`;
            statusCard.classList.add("status-healthy");

            // Refresh vector store statistics
            fetchKnowledgeBaseStats();
            fetchLLMStatus();
        } catch (error) {
            statusSpinner.classList.add("d-none");
            statusIcon.className = "bi bi-exclamation-triangle-fill text-danger fs-5";
            statusText.textContent = `Offline / Error connecting to backend (${error.message})`;
            statusCard.classList.add("status-error");
        }
    }

    /**
     * Fetches current vector store metrics via GET /api/rag/stats
     */
    async function fetchKnowledgeBaseStats() {
        if (!kbStatsBadge) return;
        try {
            const res = await fetch("/api/rag/stats");
            if (res.ok) {
                const data = await res.json();
                const count = data.stats ? data.stats.total_vectors : 0;
                const docs = data.stats ? data.stats.ingested_documents_count : 0;
                kbStatsBadge.textContent = `FAISS Vectors: ${count} (${docs} doc${docs === 1 ? '' : 's'})`;
            }
        } catch {
            // Ignore background stats fetch errors
        }
    }

    /**
     * Fetches LLM provider status via GET /api/llm/status
     */
    async function fetchLLMStatus() {
        if (!wfLlmBadge) return;
        try {
            const res = await fetch("/api/llm/status");
            if (res.ok) {
                const data = await res.json();
                if (data.configured) {
                    wfLlmBadge.className = "badge bg-success-subtle text-success border px-3 py-2 rounded-pill font-monospace small";
                    wfLlmBadge.textContent = `LLM: ${data.provider} (${data.model})`;
                } else {
                    wfLlmBadge.className = "badge bg-secondary-subtle text-secondary border px-3 py-2 rounded-pill font-monospace small";
                    wfLlmBadge.textContent = `LLM: ${data.provider}`;
                }
            }
        } catch {
            // Ignore background errors
        }
    }

    /**
     * Escapes HTML strings to prevent XSS injection
     */
    function escapeHtml(str) {
        if (!str) return "";
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    function showAlert(element, type, htmlContent) {
        if (!element) return;
        element.className = `alert alert-${type} mt-3 mb-0`;
        element.innerHTML = htmlContent;
        element.classList.remove("d-none");
    }

    function hideAlert(element) {
        if (!element) return;
        element.classList.add("d-none");
        element.innerHTML = "";
    }

    /**
     * Renders academic paper cards in the DOM (Step 2 OpenAlex)
     */
    function renderPapers(papers, topic, count) {
        if (!papersList) return;
        papersList.innerHTML = "";

        if (count === 0 || papers.length === 0) {
            resultsSection.classList.add("d-none");
            showAlert(searchAlert, "info", `No academic papers were found matching "<strong>${escapeHtml(topic)}</strong>". Try broader research keywords.`);
            return;
        }

        resultsTitle.textContent = `Retrieved Scholarly Papers (${papers.length})`;
        resultsSubtitle.textContent = `Topic: "${topic}" • Stored & deduplicated in SQLite`;
        resultsSection.classList.remove("d-none");

        papers.forEach((paper, index) => {
            const card = document.createElement("div");
            card.className = "card border-0 shadow-sm rounded-3 p-4 bg-white paper-card";

            const title = escapeHtml(paper.title || "Untitled Paper");
            const authors = escapeHtml(paper.authors || "Unknown Authors");
            const year = paper.published_year ? `<span class="badge bg-light text-dark border me-2">${paper.published_year}</span>` : "";
            const openAlexId = paper.openalex_id ? escapeHtml(paper.openalex_id) : "";
            const landingUrl = paper.url || paper.doi || openAlexId;
            const doi = paper.doi ? escapeHtml(paper.doi) : "";
            const pdfUrl = paper.pdf_url;

            let abstractHtml = "";
            if (paper.abstract && paper.abstract.trim()) {
                abstractHtml = `
                    <div class="mt-3 p-3 bg-light rounded-2 border-start border-3 border-primary">
                        <small class="fw-semibold text-secondary d-block mb-1">Abstract:</small>
                        <p class="text-secondary small mb-0 paper-abstract">${escapeHtml(paper.abstract)}</p>
                    </div>
                `;
            } else {
                abstractHtml = `
                    <div class="mt-2 py-2 px-3 bg-light rounded text-muted extra-small">
                        <i class="bi bi-info-circle me-1"></i> Abstract not provided in publisher metadata.
                    </div>
                `;
            }

            let pdfActionHtml = "";
            if (pdfUrl) {
                pdfActionHtml = `
                    <a href="${escapeHtml(pdfUrl)}" target="_blank" rel="noopener noreferrer" class="btn btn-sm btn-outline-danger d-inline-flex align-items-center gap-1">
                        <i class="bi bi-file-earmark-pdf"></i> Open Access PDF
                    </a>
                `;
            } else {
                pdfActionHtml = `
                    <span class="badge text-bg-light text-muted border py-2 px-2 small fw-normal">
                        <i class="bi bi-file-earmark-lock me-1"></i> Full-text PDF not directly accessible
                    </span>
                `;
            }

            let landingBtnHtml = "";
            if (landingUrl) {
                landingBtnHtml = `
                    <a href="${escapeHtml(landingUrl)}" target="_blank" rel="noopener noreferrer" class="btn btn-sm btn-outline-primary d-inline-flex align-items-center gap-1">
                        <i class="bi bi-box-arrow-up-right"></i> View Article
                    </a>
                `;
            }

            let doiHtml = "";
            if (doi) {
                doiHtml = `<span class="badge bg-light text-secondary border me-2 small">DOI: ${doi}</span>`;
            }

            card.innerHTML = `
                <div class="d-flex align-items-start justify-content-between gap-3 mb-2">
                    <h6 class="fw-bold text-navy mb-0 flex-grow-1">
                        <span class="text-muted fw-normal me-2">#${index + 1}</span>
                        ${landingUrl ? `<a href="${escapeHtml(landingUrl)}" target="_blank" class="text-decoration-none text-navy paper-title-link">${title}</a>` : title}
                    </h6>
                </div>
                
                <div class="text-muted small mb-2">
                    <i class="bi bi-person-fill me-1 text-secondary"></i>${authors}
                </div>

                <div class="d-flex flex-wrap align-items-center gap-1 mb-2">
                    ${year}
                    ${doiHtml}
                    ${paper.id ? `<span class="badge bg-light text-muted border small">DB ID: ${paper.id}</span>` : ""}
                </div>

                ${abstractHtml}

                <div class="d-flex flex-wrap align-items-center justify-content-between gap-2 mt-3 pt-2 border-top">
                    <div class="d-flex align-items-center gap-2">
                        ${pdfActionHtml}
                        ${landingBtnHtml}
                    </div>
                    ${openAlexId ? `<a href="${openAlexId}" target="_blank" class="text-muted extra-small text-decoration-none"><i class="bi bi-link-45deg"></i> OpenAlex Work</a>` : ""}
                </div>
            `;

            papersList.appendChild(card);
        });
    }

    /**
     * Handles OpenAlex literature search submission
     */
    async function handleSearch(event) {
        event.preventDefault();
        const topic = topicInput.value.trim();
        const limit = limitSelect ? parseInt(limitSelect.value, 10) : 10;

        if (!topic) {
            showAlert(searchAlert, "warning", "Please provide a research topic before submitting.");
            topicInput.focus();
            return;
        }

        searchBtn.disabled = true;
        searchBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Retrieving Papers...`;
        hideAlert(searchAlert);

        try {
            const response = await fetch("/api/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ topic: topic, limit: limit })
            });

            const result = await response.json();

            if (response.ok && result.status === "success") {
                showAlert(searchAlert, "success", `<strong>Success:</strong> ${escapeHtml(result.message)}`);
                renderPapers(result.papers || [], topic, result.count || 0);
            } else {
                showAlert(searchAlert, "danger", result.message || "Failed to retrieve papers.");
                if (resultsSection) resultsSection.classList.add("d-none");
            }
        } catch (error) {
            showAlert(searchAlert, "danger", `Connection error: ${error.message}. Ensure Flask backend is running.`);
            if (resultsSection) resultsSection.classList.add("d-none");
        } finally {
            searchBtn.disabled = false;
            searchBtn.innerHTML = `<i class="bi bi-search me-1"></i> Search Papers`;
        }
    }

    /**
     * Handles PDF file upload (POST /api/upload)
     */
    async function handleUpload() {
        if (!pdfFilesInput.files || pdfFilesInput.files.length === 0) {
            showAlert(uploadAlert, "warning", "Please select at least one PDF file to upload.");
            return;
        }

        const formData = new FormData();
        for (let i = 0; i < pdfFilesInput.files.length; i++) {
            formData.append("files", pdfFilesInput.files[i]);
        }

        uploadBtn.disabled = true;
        uploadBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Uploading...`;
        hideAlert(uploadAlert);

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.status === "success") {
                showAlert(uploadAlert, "success", `<strong>Upload Complete:</strong> ${escapeHtml(data.message)}`);
                if (data.files && data.files.length > 0) {
                    currentUploadedFilename = data.files[0].filename;
                    uploadedFilenameBadge.textContent = currentUploadedFilename;
                    uploadedFilesBox.classList.remove("d-none");
                    ingestBtn.disabled = false;
                }
            } else {
                showAlert(uploadAlert, "danger", data.message || "PDF upload failed.");
                ingestBtn.disabled = true;
            }
        } catch (error) {
            showAlert(uploadAlert, "danger", `Upload error: ${error.message}`);
            ingestBtn.disabled = true;
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = `<i class="bi bi-cloud-arrow-up-fill me-1"></i> 1. Upload to Server`;
        }
    }

    /**
     * Handles PDF ingestion into FAISS vector store (POST /api/ingest)
     */
    async function handleIngest() {
        if (!currentUploadedFilename) {
            showAlert(uploadAlert, "warning", "No uploaded PDF selected for ingestion.");
            return;
        }

        ingestBtn.disabled = true;
        ingestBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Ingesting into FAISS...`;
        hideAlert(uploadAlert);

        try {
            const response = await fetch("/api/ingest", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: currentUploadedFilename })
            });

            const data = await response.json();

            if (response.ok && (data.status === "success" || data.status === "already_indexed")) {
                const badgeType = data.status === "already_indexed" ? "info" : "success";
                showAlert(
                    uploadAlert,
                    badgeType,
                    `<strong>${data.status === "already_indexed" ? "Notice:" : "Ingestion Complete:"}</strong> ${escapeHtml(data.message)}<br>` +
                    `<small class="d-block mt-1">Indexed ${data.chunks_indexed || 0} chunk(s). Total vectors in FAISS store: ${data.total_vectors_in_store || 'updated'}.</small>`
                );
                fetchKnowledgeBaseStats();
            } else {
                showAlert(uploadAlert, "warning", `<strong>Ingestion Notice:</strong> ${escapeHtml(data.message || 'Failed to ingest PDF.')}`);
            }
        } catch (error) {
            showAlert(uploadAlert, "danger", `Ingestion error: ${error.message}`);
        } finally {
            ingestBtn.disabled = false;
            ingestBtn.innerHTML = `<i class="bi bi-database-fill-add me-1"></i> 2. Ingest Uploaded PDF into FAISS`;
        }
    }

    /**
     * Renders retrieved RAG passages
     */
    function renderPassages(passages, query, count) {
        if (!passagesList) return;
        passagesList.innerHTML = "";

        if (count === 0 || passages.length === 0) {
            passagesSection.classList.add("d-none");
            showAlert(retrieveAlert, "info", "No relevant passages found in the knowledge base for this question.");
            return;
        }

        passagesTitle.textContent = `Retrieved Evidence Passages (${passages.length})`;
        passagesSubtitle.textContent = `Query: "${query}" • Ranked by Cosine Similarity`;
        passagesSection.classList.remove("d-none");

        passages.forEach((p, index) => {
            const card = document.createElement("div");
            card.className = "card border-0 shadow-sm rounded-3 p-3 bg-white passage-card";

            const score = p.relevance_score != null ? p.relevance_score : 0.0;
            let scoreBadgeClass = "bg-success";
            if (score < 0.3) scoreBadgeClass = "bg-secondary";
            else if (score < 0.5) scoreBadgeClass = "bg-primary";

            const scorePercent = (score * 100).toFixed(1);
            const filename = escapeHtml(p.filename || "Document");
            const pageNum = p.page_number ? `<span class="badge bg-light text-dark border me-2">Page ${p.page_number}</span>` : "";
            const doi = p.doi ? `<span class="badge bg-light text-muted border me-2 extra-small">DOI: ${escapeHtml(p.doi)}</span>` : "";

            card.innerHTML = `
                <div class="d-flex align-items-center justify-content-between mb-2">
                    <div class="d-flex align-items-center gap-2">
                        <span class="badge bg-navy text-white">#${index + 1}</span>
                        <strong class="text-navy small">${filename}</strong>
                    </div>
                    <div class="d-flex align-items-center gap-2">
                        ${pageNum}
                        <span class="badge ${scoreBadgeClass}">Similarity: ${scorePercent}%</span>
                    </div>
                </div>

                <div class="p-3 bg-light rounded-2 border-start border-3 border-success mb-2">
                    <p class="text-secondary small mb-0 passage-text">${escapeHtml(p.passage_text)}</p>
                </div>

                <div class="d-flex align-items-center justify-content-between extra-small text-muted">
                    <div>${doi}</div>
                    <span>Chunk ID: <code>${escapeHtml(p.chunk_id || 'N/A')}</code></span>
                </div>
            `;

            passagesList.appendChild(card);
        });
    }

    /**
     * Handles RAG Passage Retrieval with Dense, Sparse, or Hybrid fusion
     */
    async function handleRetrieve(event) {
        event.preventDefault();
        const query = ragQueryInput.value.trim();
        const topK = topKSelect ? parseInt(topKSelect.value, 10) : 5;
        const mode = retrievalModeSelect ? retrievalModeSelect.value : "hybrid";

        if (!query) {
            showAlert(retrieveAlert, "warning", "Please enter a research question or topic.");
            ragQueryInput.focus();
            return;
        }

        retrieveBtn.disabled = true;
        retrieveBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Retrieving...`;
        hideAlert(retrieveAlert);

        try {
            let endpoint = "/api/retrieve/hybrid";
            let payload = { query: query, top_k: topK };

            if (mode === "dense") {
                endpoint = "/api/retrieve";
            }

            const response = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (response.ok) {
                if (retrievalModeBadge) {
                    if (mode === "dense") {
                        retrievalModeBadge.textContent = "FAISS Dense (Cosine)";
                    } else if (mode === "bm25") {
                        retrievalModeBadge.textContent = "BM25 Okapi (Keyword)";
                    } else {
                        retrievalModeBadge.textContent = "Hybrid Fusion (RRF k=60 + Re-ranking)";
                    }
                }

                const passages = data.passages || [];

                if (data.status === "empty_knowledge_base" || passages.length === 0) {
                    showAlert(retrieveAlert, "warning", `<strong>Knowledge Base Notice:</strong> ${escapeHtml(data.message || 'No passages found for this question.')}`);
                    passagesSection.classList.add("d-none");
                } else if (data.status === "insufficient_evidence") {
                    showAlert(retrieveAlert, "info", escapeHtml(data.message));
                    passagesSection.classList.add("d-none");
                } else {
                    const alertType = data.status === "low_confidence" ? "warning" : "success";
                    const msg = data.message || `Retrieved ${passages.length} evidence passages using ${mode} retrieval.`;
                    showAlert(retrieveAlert, alertType, escapeHtml(msg));
                    renderPassages(passages, query, passages.length);
                }
            } else {
                showAlert(retrieveAlert, "danger", data.message || "Failed to retrieve evidence passages.");
                passagesSection.classList.add("d-none");
            }
        } catch (error) {
            showAlert(retrieveAlert, "danger", `Retrieval connection error: ${error.message}`);
            passagesSection.classList.add("d-none");
        } finally {
            retrieveBtn.disabled = false;
            retrieveBtn.innerHTML = `<i class="bi bi-search me-1"></i> Retrieve Evidence`;
        }
    }

    /**
     * Markdown to HTML converter for client-side literature review rendering
     */
    function renderMarkdownToHtml(md) {
        if (!md) return "";
        let html = escapeHtml(md);

        // Headings
        html = html.replace(/^### (.*$)/gim, '<h3 class="mt-3 mb-2 fw-bold text-navy">$1</h3>');
        html = html.replace(/^## (.*$)/gim, '<h2 class="mt-4 mb-2 fw-bold text-navy border-bottom pb-1">$1</h2>');
        html = html.replace(/^# (.*$)/gim, '<h1 class="mt-3 mb-3 fw-bold text-dark border-bottom pb-2">$1</h1>');

        // Bold and italics
        html = html.replace(/\*\*\*(.*?)\*\*\*/gim, '<strong><em>$1</em></strong>');
        html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
        html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');

        // Blockquotes
        html = html.replace(/^\&gt;\s?(.*$)/gim, '<blockquote class="border-start border-3 border-primary ps-3 py-1 my-2 text-secondary fst-italic">$1</blockquote>');

        // Unordered lists
        html = html.replace(/^\- (.*$)/gim, '<li class="mb-1">$1</li>');
        html = html.replace(/^\* (.*$)/gim, '<li class="mb-1">$1</li>');
        html = html.replace(/(<li.*<\/li>)/gim, '<ul class="ps-3 mb-2">$1</ul>');
        html = html.replace(/<\/ul>\s*<ul class="ps-3 mb-2">/gim, '');

        // Inline code
        html = html.replace(/`([^`]+)`/gim, '<code class="bg-light px-1 py-0.5 rounded text-danger">$1</code>');

        // Split paragraphs
        const paragraphs = html.split(/\n\n+/);
        html = paragraphs.map(p => {
            const trimmed = p.trim();
            if (!trimmed) return "";
            if (trimmed.startsWith("<h") || trimmed.startsWith("<ul") || trimmed.startsWith("<blockquote") || trimmed.startsWith("<table")) {
                return trimmed;
            }
            return `<p class="mb-3 text-secondary lh-lg">${trimmed.replace(/\n/g, "<br>")}</p>`;
        }).join("\n");

        return html;
    }

    /**
     * Stepper pill state updater
     */
    function updateStepper(activeStep, maxStep = 11) {
        for (let i = 1; i <= maxStep; i++) {
            const pill = document.getElementById(`step-pill-${i}`);
            if (!pill) continue;
            pill.classList.remove("step-active", "step-done");
            if (i < activeStep) {
                pill.classList.add("step-done");
            } else if (i === activeStep) {
                pill.classList.add("step-active");
            }
        }
    }

    /**
     * Renders Cross-Paper Comparison Matrix Table
     */
    function renderComparisonMatrix(matrix) {
        if (!wfMatrixTbody) return;
        wfMatrixTbody.innerHTML = "";

        if (!matrix || matrix.length === 0) {
            wfMatrixTbody.innerHTML = `<tr><td colspan="8" class="text-center text-muted p-4">No comparison records extracted.</td></tr>`;
            return;
        }

        matrix.forEach(row => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>
                    <strong class="text-navy">${escapeHtml(row.title || 'Untitled')}</strong>
                    <div class="mt-1"><span class="badge bg-light text-dark border extra-small">${row.year || 'N/A'}</span></div>
                </td>
                <td><small class="text-secondary">${escapeHtml(row.objective || 'Not reported')}</small></td>
                <td><span class="badge bg-light text-primary border extra-small">${escapeHtml(row.methodology || 'Not reported')}</span></td>
                <td><small class="text-secondary">${escapeHtml(row.dataset || 'Not reported')}</small></td>
                <td><code>${escapeHtml(row.model || 'Not reported')}</code></td>
                <td><small class="text-secondary">${escapeHtml(row.metrics || 'Not reported')}</small></td>
                <td><small class="text-secondary">${escapeHtml(row.findings || 'Not reported')}</small></td>
                <td><small class="text-danger">${escapeHtml(row.limitations || 'Not reported')}</small></td>
            `;
            wfMatrixTbody.appendChild(tr);
        });
    }

    /**
     * Renders Thematic Clustering Cards
     */
    function renderThematicClusters(clusters) {
        if (!wfClustersContainer) return;
        wfClustersContainer.innerHTML = "";

        if (!clusters || clusters.length === 0) {
            wfClustersContainer.innerHTML = `<div class="col-12 text-center text-muted p-4">No thematic clusters formed.</div>`;
            return;
        }

        clusters.forEach(c => {
            const col = document.createElement("div");
            col.className = "col-md-6";
            const papersListHtml = (c.papers || []).map(p => `
                <li class="mb-1 text-truncate">
                    <i class="bi bi-file-earmark-text text-primary me-1"></i>
                    <span title="${escapeHtml(p.title || '')}">${escapeHtml(p.title || 'Paper')}</span>
                    ${p.published_year ? `<span class="badge bg-light text-muted border extra-small ms-1">(${p.published_year})</span>` : ''}
                </li>
            `).join("");

            col.innerHTML = `
                <div class="card border-0 shadow-sm rounded-3 p-3 bg-light h-100">
                    <div class="d-flex align-items-center justify-content-between mb-2">
                        <span class="badge bg-navy text-white">Cluster ${c.cluster_id + 1}</span>
                        <span class="badge bg-white text-secondary border font-monospace">${c.size || (c.papers || []).length} papers</span>
                    </div>
                    <h6 class="fw-bold text-navy mb-2">${escapeHtml(c.label || 'Theme')}</h6>
                    <ul class="list-unstyled extra-small text-secondary mb-0">
                        ${papersListHtml || '<li class="text-muted">No papers assigned</li>'}
                    </ul>
                </div>
            `;
            wfClustersContainer.appendChild(col);
        });
    }

    /**
     * Renders Research Gaps Cards
     */
    function renderResearchGaps(gaps) {
        if (!wfGapsContainer) return;
        wfGapsContainer.innerHTML = "";

        if (!gaps || gaps.length === 0) {
            wfGapsContainer.innerHTML = `<div class="text-center text-muted p-4">No specific research gaps detected from available literature.</div>`;
            return;
        }

        gaps.forEach((g, idx) => {
            const item = document.createElement("div");
            item.className = "card border-0 shadow-sm rounded-3 p-3 bg-light";
            item.innerHTML = `
                <div class="d-flex align-items-center justify-content-between mb-2">
                    <span class="badge bg-warning text-dark fw-semibold">Gap #${idx + 1}: ${escapeHtml(g.category || 'Research Gap')}</span>
                    <span class="badge bg-white text-muted border extra-small">Grounded in Limitations</span>
                </div>
                <h6 class="fw-bold text-dark mb-1">${escapeHtml(g.gap || '')}</h6>
                <p class="text-secondary small mb-2">${escapeHtml(g.rationale || '')}</p>
                ${g.future_direction ? `
                    <div class="p-2 bg-white rounded-2 border border-warning-subtle extra-small text-dark">
                        <strong class="text-warning-emphasis"><i class="bi bi-compass me-1"></i>Potential Future Direction:</strong>
                        ${escapeHtml(g.future_direction)}
                    </div>
                ` : ''}
            `;
            wfGapsContainer.appendChild(item);
        });
    }

    /**
     * Renders Verification Audit Cards
     */
    function renderVerificationAudit(claims) {
        if (!wfVerificationContainer) return;
        wfVerificationContainer.innerHTML = "";

        if (!claims || claims.length === 0) {
            wfVerificationContainer.innerHTML = `<div class="text-center text-muted p-4">No audited claims to verify.</div>`;
            return;
        }

        claims.forEach(c => {
            const item = document.createElement("div");
            item.className = "card border-0 shadow-sm rounded-3 p-3 bg-light";

            let badgeClass = "badge-supported";
            let icon = "bi-check-circle-fill text-success";
            const status = (c.status || "unsupported").toLowerCase();
            if (status === "partially_supported") {
                badgeClass = "badge-partially-supported";
                icon = "bi-exclamation-circle-fill text-warning";
            } else if (status === "unsupported") {
                badgeClass = "badge-unsupported";
                icon = "bi-x-circle-fill text-danger";
            } else if (status === "conflicting") {
                badgeClass = "badge-conflicting";
                icon = "bi-slash-circle-fill text-purple";
            }

            const scorePercent = c.support_score != null ? (c.support_score * 100).toFixed(0) : "N/A";

            item.innerHTML = `
                <div class="d-flex align-items-center justify-content-between mb-2">
                    <div class="d-flex align-items-center gap-2">
                        <i class="bi ${icon}"></i>
                        <strong class="small text-dark">Claim: <code>${escapeHtml(c.claim_id || 'C-1')}</code></strong>
                    </div>
                    <div class="d-flex align-items-center gap-2">
                        <span class="badge ${badgeClass} font-monospace">${escapeHtml(status.toUpperCase())}</span>
                        <span class="badge bg-white text-muted border extra-small">Grounding Score: ${scorePercent}%</span>
                    </div>
                </div>
                <p class="small text-dark fw-medium mb-2">"${escapeHtml(c.claim_text || '')}"</p>
                ${c.evidence_passage ? `
                    <div class="p-2 bg-white rounded border-start border-3 border-success extra-small text-secondary mb-2">
                        <strong class="text-dark d-block mb-1">Supporting Passage:</strong>
                        <em>"${escapeHtml(c.evidence_passage)}"</em>
                    </div>
                ` : ''}
                <div class="d-flex align-items-center justify-content-between extra-small text-muted">
                    <span>Source: <strong>${escapeHtml(c.source_paper_title || 'Literature Database')}</strong></span>
                    <span>Page: ${c.source_page != null ? c.source_page : 'N/A'}</span>
                </div>
            `;
            wfVerificationContainer.appendChild(item);
        });
    }

    /**
     * Renders Planning Agent Decomposition
     */
    function renderExecutionPlan(plan) {
        if (!wfPlanContainer) return;
        wfPlanContainer.innerHTML = "";

        if (!plan || !plan.subtopics || plan.subtopics.length === 0) {
            wfPlanContainer.innerHTML = `<div class="text-center text-muted p-4">No planning metadata recorded.</div>`;
            return;
        }

        const subtopicsHtml = plan.subtopics.map((st, i) => `
            <div class="col-md-6">
                <div class="p-3 bg-light rounded-3 border h-100">
                    <span class="badge bg-navy text-white mb-2">Subtopic ${i + 1}</span>
                    <h6 class="fw-bold text-dark small mb-1">${escapeHtml(st.subtopic || '')}</h6>
                    <p class="text-secondary extra-small mb-2"><strong>Search Query:</strong> <code>${escapeHtml(st.search_query || '')}</code></p>
                    <p class="text-muted extra-small mb-0"><strong>Evidence Criteria:</strong> ${escapeHtml(st.criteria || '')}</p>
                </div>
            </div>
        `).join("");

        wfPlanContainer.innerHTML = `
            <div class="mb-3">
                <h6 class="fw-bold text-navy mb-2">Generated Research Subtopics (${plan.subtopics.length})</h6>
                <div class="row g-3">${subtopicsHtml}</div>
            </div>
        `;
    }

    /**
     * Handles Full End-to-End Autonomous Literature Review Workflow (POST /api/workflow/run)
     */
    async function handleRunWorkflow(event) {
        event.preventDefault();
        const question = wfQuestionInput.value.trim();
        const subtopicsCount = wfSubtopicsInput ? parseInt(wfSubtopicsInput.value, 10) : 3;
        const papersPerQuery = wfPapersInput ? parseInt(wfPapersInput.value, 10) : 5;
        const maxRetries = wfRetriesInput ? parseInt(wfRetriesInput.value, 10) : 2;

        if (!question) {
            showAlert(wfAlert, "warning", "Please provide a research question or review topic.");
            wfQuestionInput.focus();
            return;
        }

        btnRunWorkflow.disabled = true;
        btnRunWorkflow.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Running Agentic Pipeline...`;
        hideAlert(wfAlert);
        wfProgressCard.classList.remove("d-none");
        wfSpinner.classList.remove("d-none");
        wfRetryBadge.textContent = `Retries: 0/${maxRetries}`;

        updateStepper(1);
        wfStatusMessage.textContent = "Stage 1/11: Decomposing research question into subtopics...";

        // Progressive stepper simulation while network request executes
        const stepTimeouts = [
            setTimeout(() => { updateStepper(2); wfStatusMessage.textContent = "Stage 2/11: Querying academic repositories & citation graph..."; }, 1500),
            setTimeout(() => { updateStepper(4); wfStatusMessage.textContent = "Stage 4/11: Indexing hybrid vector & BM25 sparse representations..."; }, 3500),
            setTimeout(() => { updateStepper(5); wfStatusMessage.textContent = "Stage 5/11: Extracting structured evidence across 8 dimensions..."; }, 5500),
            setTimeout(() => { updateStepper(6); wfStatusMessage.textContent = "Stage 6/11: Running dense KMeans thematic clustering..."; }, 7500),
            setTimeout(() => { updateStepper(7); wfStatusMessage.textContent = "Stage 7/11: Generating comparative reasoning matrix..."; }, 9500),
            setTimeout(() => { updateStepper(8); wfStatusMessage.textContent = "Stage 8/11: Identifying grounded research gaps & open problems..."; }, 11500),
            setTimeout(() => { updateStepper(9); wfStatusMessage.textContent = "Stage 9/11: Auditing claims and verifying citation grounding..."; }, 13500),
            setTimeout(() => { updateStepper(10); wfStatusMessage.textContent = "Stage 10/11: Synthesizing complete literature review report..."; }, 15500)
        ];

        try {
            const response = await fetch("/api/workflow/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: question,
                    max_subtopics: subtopicsCount,
                    papers_per_query: papersPerQuery,
                    max_retries: maxRetries
                })
            });

            stepTimeouts.forEach(t => clearTimeout(t));

            const result = await response.json();

            if (response.ok && result.status === "success") {
                const wfData = result.data || {};
                updateStepper(11);
                wfStatusMessage.textContent = "Workflow completed successfully!";
                wfSpinner.classList.add("d-none");
                wfRetryBadge.textContent = `Retries: ${wfData.retries_performed || 0}/${maxRetries}`;

                showAlert(wfAlert, "success", `<strong>Workflow Complete:</strong> Successfully executed literature review pipeline across ${wfData.papers_processed || 0} scholarly papers.`);

                // Cache report markdown
                latestReportMarkdown = wfData.report_markdown || "";

                // Populate summary meta
                if (wfSummaryMeta) {
                    wfSummaryMeta.textContent = `Papers: ${wfData.papers_processed || 0} • Passages: ${wfData.passages_indexed || 0} • Iterations: ${(wfData.retries_performed || 0) + 1} • Status: ${wfData.status}`;
                }

                // Render Subpane 1: Review Report
                if (wfReportRendered) {
                    wfReportRendered.innerHTML = renderMarkdownToHtml(latestReportMarkdown);
                }
                if (wfReportTitle) {
                    wfReportTitle.textContent = `Literature Review: "${escapeHtml(question)}"`;
                }
                if (wfReportStats) {
                    const words = latestReportMarkdown.split(/\s+/).filter(Boolean).length;
                    wfReportStats.textContent = `Word Count: ~${words} • Grounded inline citations • Generated at ${new Date().toLocaleTimeString()}`;
                }

                // Render Subpane 2: Comparison Matrix
                renderComparisonMatrix(wfData.comparison_matrix || []);

                // Render Subpane 3: Thematic Clusters
                renderThematicClusters(wfData.thematic_clusters || []);

                // Render Subpane 4: Research Gaps
                renderResearchGaps(wfData.research_gaps || []);

                // Render Subpane 5: Verification Audit
                renderVerificationAudit(wfData.verified_claims || []);

                // Render Subpane 6: Execution Plan
                renderExecutionPlan(wfData.plan || {});

                // Reveal results section
                wfResultsContainer.classList.remove("d-none");

                // Refresh vector count badge
                fetchKnowledgeBaseStats();
            } else {
                updateStepper(1);
                wfSpinner.classList.add("d-none");
                wfStatusMessage.textContent = "Workflow execution stopped.";
                showAlert(wfAlert, "danger", result.message || "Failed to execute literature review pipeline.");
            }
        } catch (error) {
            stepTimeouts.forEach(t => clearTimeout(t));
            updateStepper(1);
            wfSpinner.classList.add("d-none");
            wfStatusMessage.textContent = "Connection error during workflow.";
            showAlert(wfAlert, "danger", `Workflow error: ${error.message}. Ensure Flask backend is running.`);
        } finally {
            btnRunWorkflow.disabled = false;
            btnRunWorkflow.innerHTML = `<i class="bi bi-play-circle-fill me-1"></i> Run Complete Literature Review`;
        }
    }

    /**
     * Copy Markdown Report to Clipboard
     */
    function handleCopyReport() {
        if (!latestReportMarkdown) {
            showAlert(wfAlert, "warning", "No report available to copy.");
            return;
        }

        navigator.clipboard.writeText(latestReportMarkdown).then(() => {
            if (btnCopyReport) {
                const originalText = btnCopyReport.innerHTML;
                btnCopyReport.innerHTML = `<i class="bi bi-check2 text-success me-1"></i> Copied!`;
                setTimeout(() => {
                    btnCopyReport.innerHTML = originalText;
                }, 2000);
            }
        }).catch(err => {
            console.error("Clipboard copy failed: ", err);
        });
    }

    // Event listeners
    if (recheckBtn) recheckBtn.addEventListener("click", checkBackendHealth);
    if (searchForm) searchForm.addEventListener("submit", handleSearch);
    if (uploadBtn) uploadBtn.addEventListener("click", handleUpload);
    if (ingestBtn) ingestBtn.addEventListener("click", handleIngest);
    if (retrieveForm) retrieveForm.addEventListener("submit", handleRetrieve);
    if (wfForm) wfForm.addEventListener("submit", handleRunWorkflow);
    if (btnRunWorkflow) btnRunWorkflow.addEventListener("click", handleRunWorkflow);
    if (btnCopyReport) btnCopyReport.addEventListener("click", handleCopyReport);

    // Initial check on load
    checkBackendHealth();
});

