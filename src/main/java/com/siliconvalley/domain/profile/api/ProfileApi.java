package com.siliconvalley.domain.profile.api;

import com.siliconvalley.domain.item.myitem.application.MyItemCreateService;
import com.siliconvalley.domain.item.myitem.code.MyItemCode;
import com.siliconvalley.domain.item.myitem.dao.MyItemFindDao;
import com.siliconvalley.domain.point.application.PointManagementService;
import com.siliconvalley.domain.point.code.PointCode;
import com.siliconvalley.domain.profile.application.ProfileManagementService;
import com.siliconvalley.domain.profile.code.ProfileCode;
import com.siliconvalley.domain.profile.application.ProfilePostingService;
import com.siliconvalley.domain.profile.dao.ProfileFindDao;
import com.siliconvalley.domain.profile.dto.ProfileCreateOrUpdate;
import com.siliconvalley.domain.profile.dto.ProfileResponse;
import com.siliconvalley.global.common.code.CommonCode;
import com.siliconvalley.global.common.dto.Response;
import lombok.RequiredArgsConstructor;
import javax.validation.Valid;

import org.springframework.data.domain.Pageable;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/profiles")
@RequiredArgsConstructor
public class ProfileApi {

    private final ProfileFindDao profileFindDao;
    private final ProfileManagementService profileManagementService;
    private final MyItemFindDao myItemFindDao;
    private final MyItemCreateService myItemCreateService;
    private final PointManagementService pointManagementService;
    private final ProfilePostingService profilePostingService;

    /**
     *
     * Profile Management
     *
     * **/

    @PostMapping
    public ResponseEntity createProfile(
            @RequestBody @Valid final ProfileCreateOrUpdate dto,
            @AuthenticationPrincipal OAuth2User oAuth2User
    ) {
        String memberId = (String) oAuth2User.getAttributes().get("id");
        return ResponseEntity.status(HttpStatus.CREATED).body(profileManagementService.createProfile(memberId, dto));
    }

    @GetMapping("/{profileId}")
    public ResponseEntity getProfile(
            @PathVariable("profileId") Long profileId
    ) {
        return ResponseEntity.status(HttpStatus.OK).body(profileFindDao.getProfileById(profileId));
    }

    @PatchMapping("/{profileId}")
    public ResponseEntity updateProfile(
            @PathVariable("profileId") Long profileId,
        @RequestBody @Valid final ProfileCreateOrUpdate dto
    ) {
        return ResponseEntity.status(HttpStatus.NO_CONTENT).body(profileManagementService.updateProfile(profileId, dto));
    }

    @DeleteMapping("/{profileId}")
    public ResponseEntity deleteProfile(
        @PathVariable("profileId") Long profileId
    ) {
        return ResponseEntity.status(HttpStatus.NO_CONTENT).body(profileManagementService.deleteProfile(profileId));
    }

    /**
     *
     *  MyItem Management
     *
     * **/

    @GetMapping("/{profileId}/items")
    public ResponseEntity getItemsOfProfile(
            @PathVariable("profileId") Long profileId,
            @RequestParam(value = "category", required = false) String category,
            Pageable pageable
    ) {
        Response response = Response.of(CommonCode.GOOD_REQUEST, myItemFindDao.getMyItemListByPage(profileId, pageable, category));
        return new ResponseEntity(response, HttpStatus.OK);
    }

    @PostMapping("/{profileId}/items/{itemId}")
    public ResponseEntity purchaseSubjectItem(
            @PathVariable(name = "profileId") Long profileId,
            @PathVariable(name = "itemId") Long itemId,
            @RequestParam(value = "category") String category // subject or avatar
    ) {
        Response response = Response.of(MyItemCode.CREATE_SUCCESS, myItemCreateService.createMyItem(profileId, itemId, category));
        return new ResponseEntity(response, HttpStatus.CREATED);
    }


    /**
     *
     * Point Management
     *
     * **/

    @GetMapping("/{profileId}/points")
    public ResponseEntity getPoint(
            @PathVariable(name = "profileId") Long profileId
    ) {
        Response response = Response.of(CommonCode.GOOD_REQUEST, pointManagementService.getPoint(profileId));
        return new ResponseEntity(response, HttpStatus.OK);
    }

    @PatchMapping("/{profileId}/points/{newPointValue}")
    public ResponseEntity updatePoint(
            @PathVariable(name = "profileId") Long profileId,
            @PathVariable(name = "newPointValue") Long newPointValue
    ) {
        pointManagementService.updatePoint(profileId, newPointValue);
        Response response = Response.of(PointCode.UPDATE_SUCCESS);
        return new ResponseEntity(response, HttpStatus.NO_CONTENT);
    }


    /**
     *
     * Post Management
     *
     * **/

    @PostMapping("/{profileId}/canvases/{canvasId}/posts")
    public ResponseEntity<Response> postCanvas(
            @PathVariable Long profileId,
            @PathVariable Long canvasId
    ){
        Response response = profilePostingService.createPostForProfile(profileId, canvasId);
        return ResponseEntity.ok(response);
    }

    @DeleteMapping("/{profileId}/canvases/{canvasId}/posts")
    public ResponseEntity<Response> deletePost(
            @PathVariable Long profileId,
            @PathVariable Long canvasId
    ){
        Response response = profilePostingService.deletePostForProfile(profileId, canvasId);
        return ResponseEntity.ok(response);
    }

}
