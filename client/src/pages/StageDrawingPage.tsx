import { useEffect, useState, useRef } from 'react';
import styled from 'styled-components';
import { useRecoilValue, useRecoilState } from 'recoil';
import { UserProfileState } from '../recoil/profile/atom';
import { StageIdState } from '../recoil/stage/atom';
import { drawingSSE, disconnectDrawingSSE } from '../sse/drawingSSE';
import BlurBox from '../components/atoms/BlurBox';
import CheckingModal from '../components/organisms/CheckingModal';
import ProgressBar from '../components/atoms/ProgressTimeBar';
import ExitBox from '../components/organisms/ExitBox';
import Button from '../components/atoms/Button';
import { getLevelDetail, postDrawing, patchDrawing } from '../api/drawing';
import { ReactComponent as PencilIcon } from '../assets/image/etc/pencil.svg';
import { ReactComponent as MagicStickIcon } from '../assets/image/etc/magicStick.svg';
import { ReactComponent as LockIcon } from '../assets/image/etc/drawingLock.svg';

interface SketchList {
  sketchId: number;
  sketchImageUrl: string;
}

interface ApiResponse {
  id: number;
  stageNum: number;
  point: number;
  timeLimit: number;
  subject: {
    id: number;
    subjectName: string;
    subjectImage: string;
    sketchList: SketchList[];
  };
}

interface PostDrawingResponse {
  status: {
    httpStatus: string;
    code: number;
    message: string;
  };
  content: {
    canvasId: number;
    topPost: string;
  };
}

const defaultStyle = {
  display: 'inline-block',
  borderRadius: '25px',
  marginLeft: '-50px',
};

const DrawingPageWrapper = styled.div`
  width: 100vw;
  height: 100vh;
  min-height: 100vh;
  background-color: ${(props) => props.theme.stageColors.lightGreen};
  position: fixed;
  overflow: hidden;
`;

const TopWrapper = styled.div`
  display: flex;
  justify-content: space-between;
  margin-right: 30px;
  margin-left: 5px;
`;

const CanvasWrapper = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
  margin-left: auto;
  margin-right: auto;
`;

const BtnFloating = styled.div`
  z-index: 200;
`;

const BottomWrapper = styled.div`
  display: flex;
  flex-direction: row-reverse;
  margin-top: 10px;
  margin-right: 50px;
`;

const ToolWrapper = styled.div`
  margin-right: 100px;
  position: fixed;
  bottom: -230px;
  left: 0px;
  display: flex;
`;

const Lock = styled(LockIcon)`
  width: 100px;
  height: 100px;
  position: absolute;
  top: 20px;
  left: -20px;
  /* z-index: 1; */
`;

const Pencil = styled(PencilIcon)`
  width: 300px;
  height: 300px;
  cursor: pointer;
  transition: transform 0.2s;
  &:hover {
    transform: translateY(-15px);
  }
`;

const Eraser = styled(PencilIcon)`
  width: 300px;
  height: 300px;
  cursor: pointer;
  transform: scaleY(-1);
  transition: transform 0.2s;
  &:hover {
    transform: scaleY(-1) translateY(15px);
  }
`;

const MagicStick = styled(MagicStickIcon)`
  width: 320px;
  height: 320px;
  cursor: pointer;
  transition: transform 0.2s;
  &:hover {
    transform: translateY(-15px);
  }
`;

const BtnWrapper = styled.div`
  display: flex;
  gap: 10px;
`;

const SketchImage = styled.img`
  position: absolute;
  width: 500px;
  height: 500px;
  border-radius: 25px;
  margin-left: -50px;
  z-index: -100;
`;

const TransformedImage = styled.img`
  width: 500px;
  height: 500px;
  background-color: white;
  border-radius: 25px;
  margin-left: -50px;
  z-index: -100;
`;

const TransformedImageBox = styled.img`
  width: 500px;
  height: 500px;
  background-color: white;
  border-radius: 25px;
  margin-left: -50px;
  z-index: -100;
`;

// const eraserCursor = styled.div`
//   width: 20px;
//   height: 20px;
//   border-radius: 50%;
//   border: 0.5px solid black;
//   position: absolute;
//   z-index: 1;
//   pointer-events: none;
// `;

function StageDrawingPage() {
  const stageIdState = useRecoilValue(StageIdState);
  const { currentStageId } = stageIdState;
  const profileState = useRecoilValue(UserProfileState);
  const [data, setData] = useState<ApiResponse | undefined>();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [ctx, setCtx] = useState<CanvasRenderingContext2D | undefined>();
  const [array, setArray] = useState<{ x: number; y: number }[]>([]);
  const [isDrawing, setIsDrawing] = useState<boolean>(false); // 그리기 여부를 추정 (마우스 클릭중인지 여부)
  const [isLocked, setIsLocked] = useState<boolean>(false);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [isFirstTransform, setIsFirstTransform] = useState<boolean>(true);
  const [changeModalData, setChangeModalData] = useState<PostDrawingResponse>();
  const [canvasUrl, setCanvasUrl] = useState<string>('');
  const [isDrawingMode, setIsDrawingMode] = useState<boolean>(true); // 그리기 모드, 지우기 모드 구분
  const cursorRef = useRef<HTMLDivElement | null>(null);

  console.log('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$데이터', data);
  useEffect(() => {
    const getData = async () => {
      try {
        if (currentStageId !== null) {
          const response = await getLevelDetail(currentStageId);
          setData(response.content);
          console.log('스테이지 단건조회', response);
        } else {
          console.error('currentStageId가 null');
        }
      } catch (error) {
        console.error(error);
      }
    };
    getData();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const context = canvas.getContext('2d');
      if (context) {
        context.fillStyle = 'transparent'; // 캔버스 배경색을 투명으로 설정
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.lineJoin = 'round';
        context.lineWidth = 3;
        context.strokeStyle = 'black';
        setCtx(context);
      }
    }
  }, [data]);

  const canvasEventListener = (
    event: React.MouseEvent<HTMLCanvasElement>,
    type: string,
  ) => {
    if (isLocked) {
      return; // Lock 상태에서는 그림을 그리지 않음
    }

    const x = event.nativeEvent.offsetX;
    const y = event.nativeEvent.offsetY;

    if (isDrawingMode && ctx) {
      // 그리기 모드 일 때는 그리기
      ctx.globalCompositeOperation = 'source-over';
      ctx.lineWidth = 2;
      if (type === 'down') {
        setIsDrawing(true); // 마우스 클릭 시작
        array.push({ x, y });
      } else if (type === 'move' && isDrawing) {
        ctx?.save();
        ctx?.beginPath();
        ctx?.moveTo(array[array.length - 1].x, array[array.length - 1].y);
        ctx?.lineTo(x, y);
        ctx?.closePath();
        ctx?.stroke();
        ctx?.restore();
        array.push({ x, y });
      } else if (type === 'leave' || type === 'up') {
        setIsDrawing(false); // 마우스 클릭 종료
      }
    }

    if (!isDrawingMode && ctx) {
      // 지우기 모드일 때는 지우기
      ctx.globalCompositeOperation = 'destination-out'; // 지우기 모드로 설정
      ctx.lineWidth = 40;
      if (type === 'down') {
        setIsDrawing(true);
        array.push({ x, y });
      } else if (type === 'move' && isDrawing) {
        ctx?.beginPath();
        ctx?.moveTo(array[array.length - 1].x, array[array.length - 1].y);
        ctx?.lineTo(x, y);
        ctx?.closePath();
        ctx?.stroke();
        array.push({ x, y });
      } else if (type === 'leave' || type === 'up') {
        setIsDrawing(false);
      }
    }
  };

  console.log('현재 무슨 모드?', isDrawingMode);

  const handleClearCanvas = () => {
    if (ctx && canvasRef.current) {
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      setArray([]);
    }
  };

  const handleChange = async () => {
    // 화면 잠금, 변환 중 모달 오픈
    setIsLocked(true);
    setIsModalOpen(true);

    if (canvasRef.current && data) {
      const imageDataURL = canvasRef.current.toDataURL('image/png');

      // 이미지 데이터 확인을 위한 img 요소 생성
      const imageElement = new Image();
      imageElement.src = imageDataURL;

      // 이미지를 콘솔에 출력하여 확인
      imageElement.onload = () => {
        console.log('ImageData:', imageDataURL);
      };
      const convertedImg = await fetch(imageDataURL);
      const blob = await convertedImg.blob();

      let response: PostDrawingResponse;

      if (isFirstTransform) {
        // 첫 번째 변환, postDrawing 사용
        const formData = new FormData();
        formData.append('sketchFile', blob, 'drawing.jpg');
        formData.append('profileId', String(profileState.profileId)); // 숫자를 문자열로 변환
        formData.append('subjectId', String(data?.subject.id));

        formData.forEach(function (value, key) {
          console.log(`${key}: ${value}`);
        });

        response = await postDrawing(
          profileState.profileId,
          data.subject.id,
          formData,
        );
        setIsFirstTransform(false);
      } else if (changeModalData) {
        // 이후 변환, patchDrawing 사용
        const formData = new FormData();
        formData.append('sketchFile', blob, 'drawing.jpg');

        formData.forEach(function (value, key) {
          console.log(`${key}: ${value}`);
        });

        response = await patchDrawing(
          profileState.profileId,
          changeModalData.content.canvasId,
          formData,
        );

        console.log('변환하기 후 응답!', response);
        setChangeModalData(response);
      }
    }

    // SSE 연결 함수 호출
    drawingSSE(profileState.profileId, setIsModalOpen, setCanvasUrl);

    // SSE 연결 해제 함수 호출
    // disconnectDrawingSSE();
  };

  console.log('프로필', profileState);

  const handleToggleEdit = () => {
    setIsLocked(false);
  };

  const handlePencil = () => {
    setIsDrawingMode(true);
  };

  const handleEraser = () => {
    setIsDrawingMode(false);
  };

  console.log('변환 데이터', changeModalData);
  return (
    <DrawingPageWrapper>
      {isModalOpen && (
        <>
          <BlurBox />
          <CheckingModal imgPath={changeModalData?.content.topPost} />
        </>
      )}
      {data && data.subject.sketchList.length > 0 && (
        <>
          <ProgressBar durationInSeconds={data?.timeLimit} />
          <TopWrapper>
            <ExitBox color="dark" />
            <Button buttonText="완성 !" color="salmon" />
          </TopWrapper>
          <CanvasWrapper>
            <div
              className="container"
              style={{
                position: 'relative',
                marginRight: '-50px',
                padding: '0px',
              }}
            >
              <Lock style={{ display: isLocked ? 'block' : 'none' }} />
              <div
                style={{
                  position: 'absolute',
                  width: '500px',
                  height: '500px',
                  backgroundColor: 'white',
                  borderRadius: '25px',
                  marginLeft: '-50px',
                  zIndex: -200,
                  padding: '0px',
                }}
              />
              <SketchImage
                src={data.subject.sketchList[0].sketchImageUrl || ''}
                alt="이미지"
                style={{ display: isDrawing ? 'none' : 'block' }}
              />
              <canvas
                ref={canvasRef}
                width={500}
                height={500}
                style={{
                  ...defaultStyle,
                  cursor: !isDrawingMode
                    ? 'url("../assets/image/etc/eraserCursor.svg") 10 10, auto'
                    : 'auto',
                }}
                onMouseDown={(event) => {
                  canvasEventListener(event, 'down');
                  // 마우스 클릭 시 이미지 요소 숨기기
                }}
                onMouseMove={(event) => {
                  canvasEventListener(event, 'move');
                }}
                onMouseLeave={(event) => {
                  canvasEventListener(event, 'leave');
                }}
                onMouseUp={(event) => {
                  canvasEventListener(event, 'up');
                }}
              />
            </div>
            <BtnFloating>
              <Button
                buttonText="변신하기"
                color="green"
                onClick={handleChange}
              />
            </BtnFloating>
            <TransformedImage src={canvasUrl} alt="이미지" />
            {canvasUrl ? (
              <TransformedImage src={canvasUrl} alt="변환된 이미지" />
            ) : (
              <TransformedImageBox />
            )}
          </CanvasWrapper>
          <BottomWrapper>
            <ToolWrapper>
              <Pencil onClick={handlePencil} />
              <Eraser onClick={handleEraser} />
              <MagicStick />
            </ToolWrapper>
            <BtnWrapper>
              <Button
                buttonText="수정하기"
                color="lightGreen"
                onClick={handleToggleEdit}
              />
              <Button
                buttonText="모두 지우기"
                color="lightGreen"
                onClick={handleClearCanvas}
              />
            </BtnWrapper>
          </BottomWrapper>
        </>
      )}
    </DrawingPageWrapper>
  );
}

export default StageDrawingPage;
