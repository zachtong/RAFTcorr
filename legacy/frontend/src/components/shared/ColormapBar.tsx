const COLORMAPS: Record<string, string> = {
  turbo:
    "linear-gradient(to right, #30123b, #4662d7, #36aaf9, #1ae4b6, #72fe5e, #c8ef34, #faba39, #f66b19, #ca2a04, #7a0403)",
  viridis:
    "linear-gradient(to right, #440154, #482878, #3e4989, #31688e, #26828e, #1f9e89, #35b779, #6ece58, #b5de2b, #fde725)",
  jet:
    "linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff8800, #ff0000)",
  coolwarm:
    "linear-gradient(to right, #3b4cc0, #6788ee, #9abbff, #c9d7ef, #edd1c2, #f7a789, #e26952, #b40426)",
  plasma:
    "linear-gradient(to right, #0d0887, #5b02a3, #9c179e, #cb4679, #ed7953, #fdb42f, #f0f921)",
  inferno:
    "linear-gradient(to right, #000004, #1b0c41, #4a0c6b, #781c6d, #a52c60, #cf4446, #ed6925, #fb9b06, #f7d13d, #fcffa4)",
};

interface ColormapBarProps {
  colormap?: string;
}

export function ColormapBar({ colormap = "turbo" }: ColormapBarProps) {
  return (
    <div
      className="h-3 rounded w-full border border-[#3a3d45]"
      style={{ background: COLORMAPS[colormap] ?? COLORMAPS.turbo }}
    />
  );
}
