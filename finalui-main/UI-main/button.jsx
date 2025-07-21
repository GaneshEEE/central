import React, { useState } from "react";

function FileUploader() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [summary, setSummary] = useState("");
  const [columns, setColumns] = useState([]);
  const [rows, setRows] = useState([]);
  const [csv, setCsv] = useState("");

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    setFile(f);
    // Show preview for images
    if (f && f.type.startsWith("image/")) {
      setPreviewUrl(URL.createObjectURL(f));
    } else {
      setPreviewUrl("");
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    // Decide endpoint based on file type
    let endpoint = "";
    if (file.type.startsWith("image/")) {
      endpoint = "http://localhost:8000/create-chart";
    } else if (
      file.name.endsWith(".xlsx") ||
      file.name.endsWith(".csv")
    ) {
      endpoint = "http://localhost:8000/upload-excel";
    } else {
      alert("Unsupported file type");
      return;
    }

    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    // For Excel/CSV
    if (endpoint.includes("upload-excel")) {
      setSummary(data.summary);
      setColumns(data.columns);
      setRows(data.rows);
      setCsv(data.csv);
    }
    // For images (expects CSV in chart builder response)
    else if (endpoint.includes("create-chart")) {
      // You may need to adjust this depending on your backend's response
      if (data.csv) {
        setCsv(data.csv);
        // Parse CSV to columns/rows
        const [header, ...rowsArr] = data.csv
          .trim()
          .split("\n")
          .map((line) => line.split(","));
        setColumns(header);
        setRows(
          rowsArr.map((row) =>
            Object.fromEntries(header.map((col, i) => [col, row[i]]))
          )
        );
        setSummary("Table extracted from image.");
      } else {
        setSummary("No table found in image.");
        setColumns([]);
        setRows([]);
      }
    }
  };

  return (
    <div>
      <input
        type="file"
        accept=".xlsx,.csv,image/*"
        onChange={handleFileChange}
      />
      <button onClick={handleUpload}>Load Image / File</button>
      {previewUrl && (
        <div>
          <h4>Image Preview:</h4>
          <img src={previewUrl} alt="preview" style={{ maxWidth: 400 }} />
        </div>
      )}
      {summary && (
        <div>
          <h3>Summary</h3>
          <pre>{summary}</pre>
        </div>
      )}
      {columns.length > 0 && (
        <div>
          <h3>Extracted Table</h3>
          <table border="1">
            <thead>
              <tr>
                {columns.map((col) => (
                  <th key={col}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => (
                <tr key={idx}>
                  {columns.map((col) => (
                    <td key={col}>{row[col]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {/* Pass csv or rows/columns to your chart builder here */}
    </div>
  );
}

export default FileUploader;