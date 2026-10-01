import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { InfoTooltip } from "../InfoTooltip";

describe("InfoTooltip", () => {
  it("shows only the help icon until it is focused", () => {
    render(
      <InfoTooltip title="Threshold margin">
        <p>How far the score sits from the decision boundary.</p>
      </InfoTooltip>
    );

    expect(screen.getByLabelText("About Threshold margin")).toBeInTheDocument();
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("renders the title and body when opened", async () => {
    render(
      <InfoTooltip title="Threshold margin">
        <p>How far the score sits from the decision boundary.</p>
      </InfoTooltip>
    );

    fireEvent.focus(screen.getByLabelText("About Threshold margin"));

    const tooltip = await screen.findByRole("tooltip");
    expect(tooltip).toHaveTextContent("Threshold margin");
    expect(tooltip).toHaveTextContent("How far the score sits from the decision boundary.");
  });

  it("can keep clicks on the icon away from a clickable parent", () => {
    const onParentClick = vi.fn();
    render(
      <div onClick={onParentClick}>
        <InfoTooltip title="Inside a button" stopClickPropagation>
          <p>Body</p>
        </InfoTooltip>
      </div>
    );

    fireEvent.click(screen.getByLabelText("About Inside a button"));

    expect(onParentClick).not.toHaveBeenCalled();
  });
});
