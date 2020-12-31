import { Injectable } from '@angular/core';
import {HttpClient, HttpHeaders} from '@angular/common/http';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root'
})
export class TextService {

  constructor(private http: HttpClient) { }

  submitText(input: any) {
    let headers = {
      'text': input};

    // const url = environment.apiUrl + 'rest/next-words/';
    const url = 'https://predict-sy46lv4e6q-ue.a.run.app/predict';
    return this.http.get(url, {headers: headers});
  }

}
