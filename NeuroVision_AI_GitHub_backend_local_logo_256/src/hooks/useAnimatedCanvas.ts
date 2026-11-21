import { useEffect, useRef } from "react";

/**
 * Called every animation frame.
 * `width` and `height` are in CSS pixels; the context is already scaled for
 * high-DPI screens, so drawing code never has to think about devicePixelRatio.
 */
export type CanvasDrawFunction = (context: CanvasRenderingContext2D, width: number, height: number, now: number) => void;

/**
 * Keeps a <canvas> sized to its container and redraws it with requestAnimationFrame.
 * The canvas element is reused for the whole lifetime of the component.
 */
export function useAnimatedCanvas(draw: CanvasDrawFunction) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef(draw);
  drawRef.current = draw; // always call the newest version without restarting the loop

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;

    let cssWidth = 0;
    let cssHeight = 0;

    const resizeToContainer = () => {
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      cssWidth = canvas.clientWidth;
      cssHeight = canvas.clientHeight;
      canvas.width = Math.max(1, Math.round(cssWidth * pixelRatio));
      canvas.height = Math.max(1, Math.round(cssHeight * pixelRatio));
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    };

    const resizeObserver = new ResizeObserver(resizeToContainer);
    resizeObserver.observe(canvas);
    resizeToContainer();

    let frameId = 0;
    const renderFrame = (now: number) => {
      frameId = requestAnimationFrame(renderFrame);
      if (cssWidth === 0 || cssHeight === 0) return;
      drawRef.current(context, cssWidth, cssHeight, now);
    };
    frameId = requestAnimationFrame(renderFrame);

    return () => {
      cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
    };
  }, []);

  return canvasRef;
}
