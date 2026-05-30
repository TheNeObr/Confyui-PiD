import { app } from "../../scripts/app.js";

const PID_NODE_NAMES = new Set([
  "PiDDecodeLatent",
  "PiDDecodeLatentTiled",
  "PiDKSampler",
]);

function addResolutionWidget(nodeType) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = onNodeCreated?.apply(this, arguments);

    const existing = this.widgets?.find((widget) => widget.name === "resolution_display");
    if (existing) {
      this._pidResolutionValueEl = existing._pidResolutionValueEl ?? this._pidResolutionValueEl;
      return result;
    }

    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.alignItems = "center";
    container.style.justifyContent = "space-between";
    container.style.width = "calc(100% - 12px)";
    container.style.maxWidth = "calc(100% - 12px)";
    container.style.height = "24px";
    container.style.boxSizing = "border-box";
    container.style.padding = "0 8px";
    container.style.margin = "0 6px";
    container.style.marginTop = "-10px";
    container.style.border = "1px solid rgba(120, 136, 160, 0.35)";
    container.style.borderRadius = "12px";
    container.style.background = "rgba(15, 20, 30, 0.95)";
    container.style.color = "#cfd6e6";
    container.style.fontSize = "13px";
    container.style.gap = "8px";
    container.style.overflow = "hidden";

    const label = document.createElement("span");
    label.textContent = "resolution";
    label.style.opacity = "0.9";
    label.style.whiteSpace = "nowrap";
    label.style.flex = "0 0 auto";

    const value = document.createElement("span");
    value.textContent = "";
    value.style.flex = "1 1 auto";
    value.style.minWidth = "0";
    value.style.textAlign = "right";
    value.style.whiteSpace = "nowrap";
    value.style.overflow = "hidden";
    value.style.textOverflow = "ellipsis";

    container.append(label, value);

    const widget = this.addDOMWidget("resolution_display", "pid_resolution_display", container, {
      serialize: false,
      hideOnZoom: false,
    });
    widget.computeSize = (width) => [Math.max(120, (width ?? this.size?.[0] ?? 280) - 28), 24];
    widget._pidResolutionValueEl = value;
    this._pidResolutionValueEl = value;
    return result;
  };

  const onExecuted = nodeType.prototype.onExecuted;
  nodeType.prototype.onExecuted = function (message) {
    onExecuted?.apply(this, arguments);
    const text = Array.isArray(message?.text) ? message.text[0] : message?.text;
    const valueEl =
      this._pidResolutionValueEl ??
      this.widgets?.find((entry) => entry.name === "resolution_display")?._pidResolutionValueEl;
    if (!valueEl) {
      return;
    }
    valueEl.textContent = typeof text === "string" ? text : "";
    this.onResize?.(this.size);
  };
}

app.registerExtension({
  name: "ComfyUI.PiD.ResolutionWidget",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!PID_NODE_NAMES.has(nodeData.name)) {
      return;
    }
    addResolutionWidget(nodeType);
  },
});
