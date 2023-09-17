import styled, { keyframes } from 'styled-components';

interface LevelBtnProps {
  level: number;
  bottom: number;
  right: number;
}

const hoverAnimation = keyframes`
  0% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-15px); /* 호버 중간에 위로 올라감 */
  }
  100% {
    transform: translateY(0);
  }
`;

const LevelWrapper = styled.div`
  position: fixed;
  width: 200px;
  height: 144px;
  margin: 10px;
  cursor: pointer;

  &:hover {
    animation: ${hoverAnimation} 1s ease-in-out; /* 호버 시 애니메이션 적용 */
  }
`;

const LevelText = styled.span`
  color: #97560d;
  font-family: SBAggroB;
  font-size: 80px;
  font-style: normal;
  font-weight: 400;
  position: absolute;
  top: 35%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 3;
  text-shadow: 0px -1px 3px #23232380;
`;

const TopBtn = styled.div`
  width: 170px;
  height: 115px;
  border-radius: 50%;
  background: #fbee15;
  box-shadow:
    0px -11px 4px 0px rgba(0, 0, 0, 0.25) inset,
    5px 9px 11px 0px rgba(255, 255, 255, 0.5) inset;
  position: absolute;
  top: -5%;
  left: 7%;
  z-index: 2;
`;

const BottomBtn = styled.div`
  width: 200px;
  height: 131px;
  border-radius: 50%;
  background: #e7eaf8;
  box-shadow:
    0px -10px 4px 0px rgba(0, 0, 0, 0.25) inset,
    0px 4px 4px 0px rgba(0, 0, 0, 0.25);
  position: relative;
  z-index: 1;
`;

function LevelBtn({ level, bottom, right }: LevelBtnProps) {
  return (
    <LevelWrapper
      style={{ bottom: `${bottom || 0}px`, right: `${right || 0}px` }}
    >
      <LevelText>{level}</LevelText>
      <TopBtn />
      <BottomBtn />
    </LevelWrapper>
  );
}

export default LevelBtn;
