import styled from 'styled-components';
import { useRecoilValue } from 'recoil';
import { UserProfileState } from '../recoil/profile/atom';
import ExitBox from '../components/organisms/ExitBox';
import UserRupee from '../components/atoms/UserRupee';
import LevelBtn from '../components/atoms/LevelBtn';
import PageChangeButton from '../components/organisms/PageChangeButton';

const MapWrapper = styled.div`
  width: 100vw;
  height: 100vh;
  min-height: 100vh;
  position: fixed;
  overflow: hidden;
`;

const BlueSky = styled.div`
  width: 100%;
  height: 30%;
  background: linear-gradient(
    180deg,
    rgba(26, 112, 223, 0.8) 0%,
    rgba(105, 206, 252, 0.8) 100%
  );
  position: relative;
`;

const BigWhiteCloud = styled.div`
  width: 250px;
  height: 140px;
  border-radius: 50%;
  background: var(--white, #fff);
  box-shadow: -18px -18px 58px 0px rgba(0, 0, 0, 0.25) inset;
  z-index: 100;
  position: absolute;
  top: -40px;
  right: 40%;
`;

const SmallWhiteCloud = styled.div`
  width: 150px;
  height: 100px;
  border-radius: 50%;
  background: var(--white, #fff);
  box-shadow: -18px -18px 58px 0px rgba(0, 0, 0, 0.25) inset;
  z-index: 100;
  position: absolute;
  top: 5%;
  right: 22%;
`;

const ExitWrapper = styled.div`
  position: absolute;
  top: 3%;
`;

const LevelWrapper = styled.div`
  width: 100%;
  height: 70%;
  position: relative;
`;

const CharacterImage = styled.div<{ $bgImage: string | null }>`
  width: 300px;
  height: 300px;
  background-image: url(${(props) => props.$bgImage});
  background-size: cover;
  background-repeat: no-repeat;
  z-index: 300;
`;

const BottomToTopRoad = styled.div<{
  bottom?: number;
  right?: number;
}>`
  width: 579.819px;
  height: 48.096px;
  position: fixed;
  bottom: ${(props) => `${props.bottom || 0}px`};
  right: ${(props) => `${props.right || 0}px`};
  transform: rotate(-20deg);
  background: #c4e2a4;
`;

const TopToBottomRoad = styled.div<{
  bottom?: number;
  right?: number;
}>`
  width: 579.819px;
  height: 48.096px;
  position: fixed;
  bottom: ${(props) => `${props.bottom || 0}px`};
  right: ${(props) => `${props.right || 0}px`};
  transform: rotate(20deg);
  background: #c4e2a4;
`;

const GreenGround = styled.div`
  width: 100%;
  height: 70%;
  position: fixed;
  bottom: 0;
  z-index: -100;
  background: linear-gradient(180deg, #4ca652 0%, #8ecc51 27.6%);
`;

function StageMapPage() {
  const userProfile = useRecoilValue(UserProfileState);

  return (
    <MapWrapper>
      {/* <PageChangeButton /> */}
      <BlueSky />
      <ExitWrapper>
        <ExitBox color="light" />
      </ExitWrapper>
      <BigWhiteCloud />
      <SmallWhiteCloud />
      <CharacterImage $bgImage={userProfile.profileImg} />
      <UserRupee />
      <LevelWrapper>
        <BottomToTopRoad bottom={270} right={750} />
        <TopToBottomRoad bottom={270} right={330} />
        <BottomToTopRoad bottom={270} right={-70} />
        <LevelBtn level={1} bottom={120} right={1180} />
        <LevelBtn level={2} bottom={280} right={730} />
        <LevelBtn level={3} bottom={120} right={300} />
      </LevelWrapper>
      <GreenGround />
    </MapWrapper>
  );
}

export default StageMapPage;
