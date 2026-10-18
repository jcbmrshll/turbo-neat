// Networks with many inputs, drawn compact: the network beside an episode
// (dashboard.js layoutGrouped) and a member's line of descent (descent.js).

// networks with more inputs than this are drawn compact, their inputs in groups:
// a column of 62 inputs (F1TENTH's) at full spacing is far taller than the race
export const WIDE_INPUTS = 16;

// what an input belongs with, by its label: "lidar_3" with the lidar,
// "car_0_ahead" with car 0, "Neighbor 1: X" with neighbor 1; null for one of its
// own ("speed")
export function inputGroup(label = "") {
  const m = label.match(/^(.*?[_ ]\d+)[_ :]/) || label.match(/^(.*?)[_ ]\d+$/);
  return m ? m[1].replaceAll("_", " ") : null;
}

// inputs (in order, each with a label) in runs of one group: [{name, members}],
// name null for an input of its own
export function inputGroups(inputs, labelOf) {
  const groups = [];
  for (const n of inputs) {
    const g = inputGroup(labelOf(n));
    const last = groups.at(-1);
    if (last && g != null && last.name === g) last.members.push(n);
    else groups.push({ name: g, members: [n] });
  }
  return groups;
}
